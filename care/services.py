import base64
import csv
import io
import json
import logging
import math
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import httpx

from care.adapters.process import AlgorithmProcess, ModuleError
from care.camera import CameraHub
from care.config import EXERCISE, SIGNAL, VIDEO, RUNTIME, EXERCISES, ALGORITHM_KEYS
from care.events import Events
from care.i18n import translate
from care.inquiry import NOT_ASKED, ANSWERED, HIGH_RISK, voice_inquiry
from care.events import is_withdrawn, withdraw_status
from care.records import Records
from care.storage import Store, now, ALGORITHM_DEFAULTS, PLAN_DEFAULTS

log = logging.getLogger(__name__)

EXERCISE_KEYS = tuple(EXERCISES)
# 摄像头打不开（没插、被别的软件占用）时的统一标记，用来区分「设备问题」和「暂时失败」。
FAILED = "摄像头不可用"


def canonical_csv(data):
    text = data.decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text)))
    numeric = []
    header = None
    for row in rows:
        if not row or not any(cell.strip() for cell in row):
            continue
        try:
            values = [float(cell) for cell in row]
        except ValueError:
            if header is None and not numeric:
                header = [v.strip().lower() for v in row]
                continue
            raise ValueError("CSV 含有无法识别的数据行")
        if len(values) not in {4, 5} or not all(math.isfinite(v) for v in values):
            raise ValueError("CSV 需要 4 或 5 列有限数值")
        numeric.append(values)
    if len(numeric) < 12:
        raise ValueError("至少需要 12 个有效加速度样本")
    widths = {len(r) for r in numeric}
    if len(widths) != 1:
        raise ValueError("CSV 列数不一致")
    if len(numeric[0]) == 4:
        if not header or header[0] not in {"t", "t_s", "t_ms"}:
            raise ValueError("4 列 CSV 需要表头 t,ax,ay,az（秒）或 t_ms,ax,ay,az（毫秒）；加速度单位 g")
        factor = 1 if header[0] == "t_ms" else 1000
        numeric = [[r[0] * factor, math.sqrt(sum(v * v for v in r[1:])), *r[1:]] for r in numeric]
    if any(b[0] <= a[0] for a, b in zip(numeric, numeric[1:])):
        raise ValueError("样本时间戳必须严格递增")
    output = io.StringIO(newline="")
    csv.writer(output).writerows(numeric)
    return output.getvalue().encode("utf-8")


def _camera_partners(sid):
    """与本会话共用同一物理摄像头的其它会话；独占前需要它们先释放。"""
    if sid == "video":
        return {"exercise-opening"}
    return {"video", "exercise-opening"}


def _launcher():
    """重启时用的解释器：Windows 下优先 pythonw.exe，避免弹出控制台窗口。"""
    if os.name != "nt":
        return sys.executable
    candidate = Path(sys.executable).with_name("pythonw.exe")
    return str(candidate) if candidate.is_file() else sys.executable


def _script():
    return Path(__file__).resolve().parents[1] / "run.py"


class System:
    def __init__(self, store=None):
        self.store = store or Store()
        self.events = Events(self.store)
        self.records = Records(self.store)
        self.modules = {name: AlgorithmProcess(name) for name in ("video", "signal", "exercise", "audio")}
        self.pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="homeease")
        self.stop_event = threading.Event()
        self.video = {"state": "idle", "prob": 0, "frames": 0}
        self.video_mode = "replay"
        self.video_run = None
        self.video_finalized = True
        self.video_last_frame_at = 0
        self.video_event_keys = {}
        self.video_lock = threading.RLock()
        self.home_monitoring = {**self.store.home_monitoring(), "error": None}
        self.home_retry_at = 0
        self.home_initializing = None
        self.cameras = CameraHub()
        self.camera_shapes = {}
        self.sensor_lock = threading.Lock()
        self.audio_lock = threading.Lock()
        self.sensor = {"state": "disconnected", "last_received": None}
        self.sessions_lock = threading.RLock()
        self.session_metadata = {}
        self.session_commands = {}
        self.session_cancel = {}
        self.latest_poses = {}
        self.audio_last_received = None
        self.audio_quality = {"state": "active", "reason": "麦克风输入有效"}
        self.jobs = set()
        self.jobs_lock = threading.RLock()
        # 后台算法开关：只记录开关状态与调度结果，不参与原算法的判定。
        self.algorithms_lock = threading.RLock()
        self.algorithms = {key: item["enabled"] for key, item in self.store.algorithms().items()}
        self.remote_access = False
        # 后台常驻时自动运行的两条通道。``stop`` 指向各自的收尾函数，
        # 关闭开关时只需调用它，不必在路由层判断当前是哪种模式。
        self.exercises_lock = threading.RLock()
        self.sensor_session = {"active": False, "sid": None, "error": None, "stop": lambda reason: None}
        self.exercise_session = {"active": False, "sid": None, "key": None, "last_key": None, "error": None,
                                 "retry_at": 0, "started_at": None, "stop": lambda reason: None}
        self.resident = {"state": "off", "error": None, "started_at": None, "last_open_at": None,
                         "restarted_at": None, "exited_at": None}
        self.resident_retry_at = 0
        self.vision_retry_at = 0
        self.store.recover_coverage()
        for channel in ("vision", "accel", "audio"):
            self.store.set_coverage(channel, "unavailable", "尚未连接实时输入")
        for old in self.store.sessions():
            if old["status"] in {"running", "paused"}:
                old.update(status="interrupted", quality="service_restarted", ended_at=now())
                self.store.save_session(old)
        # 老版本只有「家庭守护」开关：升级后把它对齐到「视频跌倒检测」，
        # 保持既有的监测行为不变，不因为新增开关而默默关闭正在用的守护。
        legacy = self.store.home_monitoring()
        if legacy["enabled"] and not self.algorithms.get("vision", True):
            self.algorithms["vision"] = True
            self.store.save_algorithm("vision", True)

    def start(self):
        for name in self.modules:
            self.submit(self.initialize, name)
        self.scheduler = threading.Thread(target=self._loop, daemon=True, name="homeease-scheduler")
        self.scheduler.start()

    def submit(self, fn, *args):
        future = self.pool.submit(fn, *args)
        with self.jobs_lock:
            self.jobs.add(future)
        def done(job):
            with self.jobs_lock:
                self.jobs.discard(job)
            if not job.cancelled() and job.exception():
                log.error("Background task failed: %s", job.exception())
        future.add_done_callback(done)
        return future

    def initialize(self, name):
        module = self.modules[name]
        try:
            result = module.call("initialize", timeout=150)
            self.store.audit(name, "initialized", {"model": result.get("model") or result.get("model_id") or result.get("sound_model")})
            return result
        except Exception as exc:
            module.state = "unavailable"
            module.error = str(exc)
            self.store.audit(name, "initialization_failed", {"error": str(exc)})
            raise

    def _loop(self):
        while not self.stop_event.wait(1):
            for name, operation in (("events", self.events.tick), ("video", self.poll_video),
                                    ("algorithms", self.ensure_algorithms),
                                    ("coverage", self.check_coverage), ("sessions", self.check_sessions),
                                    ("notifications", self.deliver_webhooks), ("heartbeat", self.store.heartbeat)):
                try:
                    operation()
                except Exception as exc:
                    log.exception("%s scheduler operation failed", name)

    # ------------------------------------------------------------------ 后台算法开关
    def algorithm_channel(self, key):
        """读取一个开关；未知键返回 None，由路由层报 404。"""
        with self.algorithms_lock:
            return self.algorithms.get(key)

    def set_algorithms(self, changes):
        """只改被提交的开关，并立即生效（不必等调度下一轮）。

        开启「后台常驻」时只打开常驻本身，视频/加速度仍按各自的开关决定；
        关闭常驻等价于「自己开着的通道全部停掉」，但开关本身保留，下次再开还在。
        """
        changes = {key: bool(value) for key, value in changes.items() if value is not None}
        previous = {}
        with self.algorithms_lock:
            for key, value in changes.items():
                if key not in ALGORITHM_DEFAULTS:
                    raise KeyError(key)
                previous[key] = self.algorithms.get(key, ALGORITHM_DEFAULTS[key])
                if previous[key] != value:
                    self.store.save_algorithm(key, value)
                self.algorithms[key] = value
        try:
            if "vision" in changes and not changes["vision"]:
                self.stop_vision("视频跌倒检测已关闭", switch=True)
            if "accel" in changes and not changes["accel"]:
                self.stop_session_channel(self.sensor_session, "加速度跌倒检测已关闭")
            for key, value in changes.items():
                if key in EXERCISE_KEYS and not value:
                    self.stop_session_channel(self.exercise_session, f"{EXERCISES[key]}检测已关闭")
            if changes.get("resident"):
                self.resident["state"] = "starting"
                self.resident["error"] = None
                self.store.audit("resident", "enabled", {"remote_access": self.remote_access})
            elif "resident" in changes and not changes["resident"]:
                self.stop_vision("后台常驻已关闭", switch=True)
                self.stop_session_channel(self.sensor_session, "后台常驻已关闭")
                for key in EXERCISE_KEYS:
                    self.stop_session_channel(self.exercise_session, "后台常驻已关闭")
                self.resident.update(state="off", error=None)
                self.store.audit("resident", "disabled", {})
            self.sync_home_monitoring()
            self.ensure_algorithms()
        except Exception:
            log.exception("Could not apply algorithm switches %s", changes)
        return self.algorithm_status()

    def sync_home_monitoring(self):
        """后台常驻与「家庭守护」是同一个视频跌倒监测，保持两边一致。

        家庭页的暂停按钮、工作台的停止检测按钮都走这里，避免出现
        「开关显示开启、实际已暂停」或反之的矛盾状态。
        """
        with self.algorithms_lock:
            vision = self.algorithms.get("vision", True)
            resident = self.algorithms.get("resident", False)
        desired = bool(vision and (resident or self.home_monitoring["enabled"]))
        if desired != bool(self.home_monitoring["enabled"]):
            self.set_home_monitoring(desired, self.home_monitoring.get("camera_index", 0))
        return dict(self.home_monitoring)

    def stop_vision(self, reason, switch=False):
        """停下视频检测。

        ``switch=False`` 用于后台轮播这类内部让位：只停运行状态，绝不能顺手把
        「视频跌倒检测」开关也关掉 —— 否则镜头交出去之后，就没谁再把它拉回来了。
        只有照护者主动关开关时才传 ``switch=True``。
        """
        if self.video.get("state") != "running" and not self.home_monitoring["enabled"]:
            return
        # 注意 set_home_monitoring(False) 不会反向改 vision 开关，
        # 「关掉开关」这件事由调用方（set_algorithms）自己写库。
        self.set_home_monitoring(False, self.home_monitoring.get("camera_index", 0))
        self.store.audit("video", "stopped_by_switch", {"reason": reason, "switch": switch})

    @staticmethod
    def stop_session_channel(state, reason):
        """关掉一个后台通道：有会话就先结束它，没有就只清掉「已开启」标记。

        状态先标为不再活跃，避免调度在同一秒里又把它拉起来；真正的收尾
        （结束会话、释放摄像头）交给通道自己的 ``stop``。
        """
        stop = state.get("stop")
        state["active"] = False
        if callable(stop):
            try:
                stop(reason)
                return
            except Exception:
                log.exception("Could not stop background session %s", state.get("sid"))
        state["sid"] = None

    def ensure_algorithms(self):
        """调度：把开了开关、但当前没在运行的通道都交给原算法跑起来。

        所有启动都经过既有的 video_start / start_camera_session，输入来源、
        采样与判定完全沿用原算法；这里只决定「启不启动、什么时候启动」。
        """
        if self.stop_event.is_set():
            return
        with self.algorithms_lock:
            switches = dict(self.algorithms)
        self.ensure_resident(switches)
        if not switches.get("resident"):
            return
        self.ensure_vision(switches)
        self.ensure_sensor(switches)
        self.ensure_exercises(switches)

    def ensure_resident(self, switches):
        if not switches.get("resident"):
            if self.resident["state"] != "off":
                self.resident.update(state="off", error=None)
            return
        now_mono = time.monotonic()
        if self.resident["state"] == "exited" and now_mono - (self.resident.get("exited_at") or 0) < 15:
            return
        if now_mono < self.resident_retry_at or not (_script().is_file()):
            return
        self.resident_retry_at = now_mono + 30
        if self.resident.get("error") == FAILED:
            self.resident.update(error=None)  # 给一次重试机会：设备可能刚插回来
        # 已经能读到本机状态就只做自检，不重复拉起进程。
        self.store.heartbeat()
        self.resident.update(state="running", error=None, started_at=self.resident.get("started_at") or now())
        self.store.audit("resident", "alive", {"pid": os.getpid()})

    def ensure_vision(self, switches):
        with self.video_lock:
            if not (switches.get("resident") and switches.get("vision", True)):
                return
            if not self.modules["video"].state == "ready":
                return
            if self.video.get("state") == "running":
                return
            if self.exercise_session.get("active"):
                return  # 摄像头正被运动训练独占，训练结束后再接管
            if time.monotonic() < self.vision_retry_at:
                return
            if self.resident.get("error") == FAILED:
                self.resident.update(error=None)  # 先解除失联标记，让下一轮真正去试一次
            # 复用既有的守护启动路径：同样的输入、同样的限速重试与错误上报。
            if self.home_monitoring["enabled"]:
                self.ensure_home_monitoring()
                return
            self.vision_retry_at = time.monotonic() + 15
            try:
                if self.modules["video"].state != "ready":
                    return
                self.video_start({"type": "camera", "index": self.home_monitoring.get("camera_index", 0)}, "live")
                self.store.save_home_monitoring(True, self.home_monitoring.get("camera_index", 0))
                self.home_monitoring.update(enabled=True, error=None)
                self.store.audit("resident", "vision_started", {})
            except Exception as exc:
                self.resident["error"] = str(exc)
                self.store.audit("resident", "vision_retry", {"error": str(exc)})

    def ensure_sensor(self, switches):
        """加速度通道没有本机采集设备：后台常驻时只确认「谁在提供数据」。

        真实的三轴数据来自手机端 /sensor（浏览器把加速度推到 /ws/sensor），
        或用 CSV 回放。这里不伪造数据，只把状态与原因如实报给设置页。
        """
        state = self.sensor_session
        wanted = bool(switches.get("resident") and switches.get("accel", True))
        if state.get("active") and not wanted:
            self.stop_session_channel(state, "加速度跌倒检测已关闭")
            return
        connected = self.sensor["state"] == "connected"
        with self.exercises_lock:
            state["active"] = bool(wanted and connected)
            state["sid"] = "sensor" if state["active"] else None
            state["error"] = None if not wanted or connected else "本机没有三轴采集设备：请在手机上打开 /sensor 并保持页面在前台，数据会实时推送到这里"
        if state["active"]:
            state["stop"] = self._stop_sensor_channel
        else:
            state["stop"] = lambda reason: None

    def _stop_sensor_channel(self, reason):
        """关闭加速度检测不打断已连上的设备，只停止把它算作「后台通过」的依据。"""
        with self.exercises_lock:
            self.sensor_session.update(active=False, sid=None, stop=lambda value: None)
        self.store.audit("resident", "accel_stopped", {"reason": reason})

    def ensure_exercises(self, switches):
        if self.exercise_session.get("error") == FAILED:
            return  # 本机没有可用摄像头，后台轮播只会反复开机关机，等设备恢复再说
        state = self.exercise_session
        wanted = [key for key in EXERCISE_KEYS if switches.get(key, True)]
        if not switches.get("resident"):
            wanted = []
        if state.get("active"):
            if state.get("key") in wanted:
                return
            self.stop_session_channel(state, "该运动检测已关闭")
            return
        if wanted and state.get("error") and (state.get("retry_at") or 0) > time.monotonic():
            return
        if not wanted or self.modules["exercise"].state != "ready":
            return
        # 运动检测只接管「后台自己开着的」那路画面。若照护者正在监测工作台看实时画面，
        # 就把摄像头留给它，等视频停下后再接上，而不是抢占正在被观看的画面。
        handover = None
        if self.resident.get("error") == FAILED:
            return  # 摄像头打不开，视频和训练都别每轮重试
        if self.video.get("state") == "running" and self.video_mode == "live" and switches.get("vision", True):
            busy = self.cameras.busy("video")
            live = {sid for sid, meta in self.session_metadata.items() if meta.get("active")}
            if busy & live:
                return  # 真人正在用工作台做训练，别把他手上的会话换成后台轮播
            if not busy:
                handover = "video"  # 只有后台视频在占用，可以直接把镜头转过来
        target = self._next_exercise(wanted)
        if target is None:
            return
        try:
            result = self.start_camera_session(target, self.store.settings().get("exercise_target", 10),
                                               self.home_monitoring.get("camera_index", 0), handover=handover)
        except Exception as exc:
            state.update(active=False, sid=None, key=None, error=str(exc), retry_at=time.monotonic() + 60)
            self.store.audit("resident", "exercise_failed", {"exercise": target, "error": str(exc)})
            return
        sid = result["session_id"]
        with self.exercises_lock:
            state.update(active=True, sid=sid, key=target, error=None, retry_at=0, started_at=now())
            state["stop"] = lambda reason, sid=sid: self.end_background_exercise(sid, reason)
        self.store.audit("resident", "exercise_started", {"session_id": sid, "exercise": target,
                                                          "handover": bool(handover)})
        if handover:
            # 画面已交给训练会话，视频通道腾出镜头；训练结束后下一轮调度再接管。
            # keep_camera: 采集线程此刻正被训练会话使用，不能拆机。
            with self.video_lock:
                self.stop_vision("摄像头交由后台运动检测使用")

    def _next_exercise(self, wanted):
        """按计划顺序轮转；某动作本轮已完成，就让给下一个，让长者不用等。"""
        with self.exercises_lock:
            last = self.exercise_session.get("last_key")
        plan = self.store.exercise_plan()
        order = [item["exercise"] for item in plan["items"] if item["enabled"] and item["exercise"] in wanted]
        order = order or list(wanted)
        if last in order:
            order = order[order.index(last) + 1:] + order[:order.index(last) + 1]
        return order[0]

    def clear_stale_sessions(self):
        """清理没有活动训练、却还标着 running 的旧会话。

        这类残留来自上一次进程异常退出或调度竞态：它们没有对应的内存状态，
        既不会自己结束，又会一直挡着「开始新的训练」。
        """
        for item in self.store.sessions(mode="live"):
            sid = item["session_id"]
            if item["status"] in {"running", "paused"} and not self.session_metadata.get(sid, {}).get("active"):
                item.update(status="interrupted", quality="service_restarted", ended_at=now())
                self.store.save_session(item)
                self.store.audit("exercise", "stale_session_closed", {"session_id": sid})
                self.cameras.release(sid)

    def end_background_exercise(self, sid, reason):
        if sid not in self.session_metadata:
            return None
        # 先结束会话，再释放通道标记：反过来的话，会话万一结束失败
        # 就没有人记得它还在跑，会把摄像头一直占着。
        try:
            result = self.session_control(sid, "end")
        except Exception as exc:
            log.exception("Could not end background exercise %s", sid)
            result = {"session_id": sid, "status": "interrupted", "error": str(exc)}
        finally:
            with self.exercises_lock:
                state = self.exercise_session
                if state.get("sid") == sid:
                    state.update(active=False, sid=None, last_key=state.get("key"), error=None)
        self.store.audit("resident", "exercise_ended", {"session_id": sid, "reason": reason,
                                                        "count": result.get("count", 0)})
        return result

    def algorithm_status(self):
        """给设置页用的完整状态：开关 + 后台此刻实际在跑什么 + 做不到的原因。"""
        with self.algorithms_lock:
            switches = dict(self.algorithms)
        vision_on = self.video.get("state") == "running" and self.video_mode == "live"
        sensor = self.sensor_session
        exercise = self.exercise_session
        channels = [
            {"key": "vision", "type": "algorithm", "active": vision_on,
             "detail": self.video.get("error") or ("实时视频检测中" if vision_on else "未运行")},
            {"key": "accel", "type": "algorithm",
             "active": bool(sensor.get("active")) or self.sensor["state"] == "connected",
             "detail": sensor.get("error") or ("后台采集运行中" if sensor.get("active") else
                                               "手机端已连接" if self.sensor["state"] == "connected" else "未运行")},
        ]
        current = exercise.get("key") if exercise.get("active") else None
        for key in EXERCISE_KEYS:
            running = current == key
            channels.append({"key": key, "type": "exercise", "active": running,
                             "detail": "随后台常驻识别动作" if running else
                                       "待识别" if switches.get("resident") and switches.get(key) else "未运行"})
        reasons = self.algorithm_blockers(switches)
        return {"switches": switches, "channels": channels, "resident": dict(self.resident),
                "resident_ready": self.resident_available(), "blockers": reasons,
                "remote_access": self.remote_access}

    def resident_available(self):
        return _script().is_file()

    def sensors_connected(self):
        """加速度通道的数据来自三轴设备（手机端 /sensor），本机没有第二个采集源。"""
        return self.sensor["state"] == "connected"

    def plan_progress(self, plan=None):
        """今天各动作的完成情况；完成判定沿用既有训练记录（status=ended, mode=live）。"""
        plan = plan or self.store.exercise_plan()
        tz = ZoneInfo("Asia/Shanghai")
        today = datetime.now(tz).date()
        sessions = [s for s in self.store.sessions(mode="live", limit=2000)
                    if s.get("status") == "ended"
                    and datetime.fromisoformat(s["created_at"]).astimezone(tz).date() == today]
        items = []
        for item in plan["items"]:
            done = [s for s in sessions if s["exercise"] == item["exercise"]]
            reps = sum(s.get("count", 0) for s in done)
            target = max(1, item["target_reps"])
            items.append({**item, "reps": reps, "sessions": len(done),
                          "done": bool(done) and reps >= target,
                          "disabled": not self.exercise_enabled(item["exercise"])})
        return {"date": today.isoformat(), "items": items,
                "completed": sum(1 for i in items if i["done"] and i["enabled"]),
                "total": sum(1 for i in items if i["enabled"]),
                "skipped": today.isoformat() in (plan.get("skipped_dates") or [])}

    def save_plan(self, value, updated_by="caregiver"):
        plan = {**PLAN_DEFAULTS, **value, "updated_at": now(), "updated_by": updated_by}
        self.store.save_exercise_plan(plan)
        return {**plan, "progress": self.plan_progress(plan)}

    def shift_plan(self, delta):
        """长者端「今天不做了」：只把这次跳过记到今天为止，不动计划内容。

        不把计划本身挪到明天 —— 计划是照护者安排的每日日程，需要顺延的是
        「今天这一份」。跳过日期落库后，今天的长者端就不再显示，明天照常出现。
        """
        plan = self.store.exercise_plan()
        tz = ZoneInfo("Asia/Shanghai")
        today = datetime.now(tz).date()
        skipped = {d for d in (plan.get("skipped_dates") or []) if d >= today.isoformat()}
        for step in range(1, max(1, int(delta)) + 1):
            skipped.add((today + timedelta(days=step - 1)).isoformat())
        plan.update(skipped_dates=sorted(skipped), updated_at=now(), updated_by="elder")
        self.store.save_exercise_plan(plan)
        self.store.audit("plan", "deferred", {"dates": sorted(skipped)})
        return {**plan, "progress": self.plan_progress(plan)}

    def algorithm_blockers(self, switches):
        blockers = []
        if not switches.get("resident"):
            blockers.append("「后台常驻」未开启，视频、加速度与运动检测不会自动运行")
            return blockers
        if not self.resident_available():
            blockers.append("找不到启动脚本，后台常驻无法自动恢复")
        if self.resident.get("error") and self.resident["error"] != FAILED:
            blockers.append(self.resident["error"])
        if self.resident.get("error") and self.resident["error"] == FAILED:
            blockers.append("本机摄像头不可用，插上或释放占用后会自动恢复")
        elif switches.get("vision") and self.modules["video"].state != "ready":
            blockers.append("视频算法尚未就绪，启动后会自动接管")
        if self.exercise_session.get("error"):
            blockers.append(self.exercise_session["error"])
        return blockers

    def exercise_enabled(self, key):
        with self.algorithms_lock:
            return bool(self.algorithms.get(key, True))

    def set_remote_access(self, value):
        self.remote_access = bool(value)
        if value:
            self.store.audit("resident", "remote_access_reported", {"pid": os.getpid()})

    def video_pose(self):
        """当前帧的人体框与 17 个关键点（归一化 0–1），供前端叠加层使用。

        检测框与骨架由工作台前端在自己的画布上绘制并各自开关；这里只读原引擎
        已经算好的结果，不参与判定，也无需向算法目录暴露任何检测参数。
        """
        if self.modules["video"].state != "ready":
            raise ModuleError("视频模块尚未就绪", 503)
        if self.video.get("state") not in {"running", "idle", "finished"}:
            raise ModuleError("视频检测未在运行", 409)
        return self.modules["video"].call("pose", timeout=10)

    def video_start(self, source, mode):
        with self.video_lock:
            if self.home_monitoring["enabled"] and mode != "live":
                raise ValueError("请先暂停居家守护，再播放视频样例")
            if self.video.get("state") == "running":
                raise ValueError("视频检测正在运行")
            self.video_mode = mode
            self.video_run = uuid.uuid4().hex
            self.video_finalized = False
            self.video_event_keys = {}
            if source.get("type") == "camera":
                # 先占位再开机：acquire 返回时采集线程已经在出帧，交接时就能直接复用，
                # 万一视频模块启动失败也不会留下一路没人认领的摄像头。
                source = {**source, "shared_camera": self.cameras.acquire(source.get("index", 0), "video")}
                try:
                    self.video = self.modules["video"].call("start", source=source)
                except Exception:
                    self.cameras.release("video")
                    raise
            else:
                self.video = self.modules["video"].call("start", source=source)
            self.video_last_frame_at = time.time()
            self.store.audit("video", "started", {"mode": mode, "source": source.get("type")})
            return self.video

    def video_stop(self, keep_camera=None):
        with self.video_lock:
            try:
                self.video = self.modules["video"].call("stop", timeout=20)
            finally:
                # keep_camera: 这一路画面马上要交给后台运动检测，先不拆机。
                self.cameras.release("video", accept=[keep_camera] if keep_camera else ())
            self.video_finalized = True
            self.video_run = None
            self.poll_video(force=True)
            self.store.set_coverage("vision", "unavailable", "实时视频已停止")
            return self.video

    def poll_video(self, force=False):
        if not self.video_run or (self.video_finalized and not force):
            return
        with self.video_lock:
            previous = self.video.get("frames", 0)
            try:
                status = self.modules["video"].call("status", timeout=12)
            except Exception as exc:
                self.video.update(state="error", error=str(exc))
                if self.home_monitoring["enabled"]:
                    self.video_finalized = True
                self.cameras.release("video")
                raise
            self.video = {k: v for k, v in status.items() if k != "records"}
            self.video_finalized = status["state"] != "running" and not status.get("processing", False)
            if self.video_finalized:
                self.cameras.release("video")
            if status["frames"] != previous:
                self.video_last_frame_at = time.time()
            alarm_at = status.get("alarm_at")
            if alarm_at and alarm_at not in self.video_event_keys:
                event = self.events.create("vision", {"prob": status["prob"], "threshold": status["threshold"]}, self.video_mode,
                    source=status["source"], dedup_key=f"video:{self.video_run}:{alarm_at}",
                    at=datetime.fromtimestamp(alarm_at, timezone.utc).isoformat(timespec="milliseconds"))
                self.video_event_keys[alarm_at] = event["id"]
            for record in status.get("records", []):
                if not self.video_event_keys:
                    continue
                alarm = min(self.video_event_keys, key=lambda stamp: abs(stamp - record["epoch"]))
                if abs(alarm - record["epoch"]) > 2:
                    continue
                eid = self.video_event_keys[alarm]
                old = self.store.event(eid)
                if old["evidence"].get("clip") == record.get("clip"):
                    continue
                def add_clip(event, record=record):
                    event["evidence"].update(record)
                    Events.timeline(event, "evidence_saved", "原视频引擎已保存事件片段")
                self.store.update_event(eid, add_clip)

    def check_coverage(self):
        if self.video_mode == "live" and self.video.get("state") == "running":
            fresh = time.time() - self.video_last_frame_at < 8
            visible = self.video.get("person_visible", False)
            self.store.set_coverage("vision", "active" if fresh and visible else "reduced",
                "实时画面有效" if fresh and visible else ("画面中未检测到人体" if fresh else "视频数据已超时"))
        else:
            self.store.set_coverage("vision", "unavailable", "正在样例回放" if self.video_mode == "replay" and self.video.get("state") == "running" else "未开启实时视频")
        if self.sensor["state"] == "connected":
            fresh = time.time() - (self.sensor.get("last_received") or 0) < 5
            self.store.set_coverage("accel", "active" if fresh else "reduced", "已接收实时传感器数据" if fresh else "传感器数据已超时")
        else:
            self.store.set_coverage("accel", "unavailable", "传感器未连接")
        consent = self.store.settings()["audio_consent"]
        fresh_audio = consent and self.audio_last_received and time.time() - self.audio_last_received < 20
        self.store.set_coverage("audio", self.audio_quality["state"] if fresh_audio else "unavailable",
            self.audio_quality["reason"] if fresh_audio else ("未开启音频授权" if not consent else "未连接连续麦克风输入"))

    def status(self):
        coverage = [r for r in self.store.coverage() if r["ended_at"] is None]
        active = sum(r["state"] == "active" for r in coverage if r["channel"] in {"vision", "accel"})
        with self.algorithms_lock:
            switches = dict(self.algorithms)
        return {"modules": [m.health() for m in self.modules.values()], "video": self.video,
                "video_mode": self.video_mode, "sensor": self.sensor, "coverage": coverage,
                "cameras": self.cameras.status(), "home_monitoring": dict(self.home_monitoring),
                "coverage_state": "active" if active == 2 else "reduced" if active else "unavailable",
                "server_time": now(), "audio_consent": self.store.settings()["audio_consent"],
                "algorithms": switches,
                "resident": {"enabled": bool(switches.get("resident")), **{k: v for k, v in self.resident.items() if k != "state"},
                             "state": self.resident["state"]},
                "background": {"sensor": {"active": bool(self.sensor_session.get("active")),
                                          "error": self.sensor_session.get("error"),
                                          "connected": self.sensor["state"] == "connected"},
                               "exercise": {"active": bool(self.exercise_session.get("active")),
                                            "key": self.exercise_session.get("key"),
                                            "error": self.exercise_session.get("error")}}}

    def set_home_monitoring(self, enabled, camera_index=0):
        """开启/停止「家庭守护」。它与设置页的「视频跌倒检测」是同一个开关。"""
        if not enabled:
            # 停下来时只写运行状态，不反向关闭 vision 开关：交门给运动检测属于
            # 内部让位，关掉开关会导致训练结束后再没人把视频拉回来。
            self.store.save_home_monitoring(False, camera_index)
            with self.video_lock:
                self.home_monitoring.update(enabled=False, camera_index=camera_index, error=None)
                self.home_retry_at = 0
                if self.video.get("state") == "running" and self.video_mode == "live":
                    self.video_stop()
            self.store.audit("home", "monitoring_disabled", {"camera_index": camera_index})
            return dict(self.home_monitoring)
        with self.video_lock:
            changed_camera = self.home_monitoring["camera_index"] != camera_index
            self.store.save_home_monitoring(enabled, camera_index)
            self.home_monitoring.update(enabled=enabled, camera_index=camera_index, error=None)
            self.home_retry_at = 0
            if self.video.get("state") == "running" and (not enabled or changed_camera or self.video_mode != "live"):
                self.video_stop()
            self.store.audit("home", "monitoring_enabled" if enabled else "monitoring_disabled", {"camera_index": camera_index})
        # 「家庭守护」与设置页的「视频跌倒检测」是同一件事，开关保持同步。
        if self.algorithms.get("vision", True) != bool(enabled):
            with self.algorithms_lock:
                self.algorithms["vision"] = bool(enabled)
            self.store.save_algorithm("vision", bool(enabled))
        # Startup/recovery runs on the scheduler, never on a page-viewer lease.
        return dict(self.home_monitoring)

    def ensure_home_monitoring(self):
        with self.video_lock:
            if self.stop_event.is_set() or not self.home_monitoring["enabled"]:
                return
            if self.video.get("state") == "running" and self.video.get("processing", True):
                return
            if time.monotonic() < self.home_retry_at:
                return
            if self.home_initializing and not self.home_initializing.done():
                return
            module = self.modules["video"]
            if module.state in {"loading", "not_initialized"}:
                return
            self.home_retry_at = time.monotonic() + 15
            try:
                if module.state != "ready":
                    module.close()
                    self.home_initializing = self.submit(self.initialize, "video")
                    return
                self.video_start({"type": "camera", "index": self.home_monitoring["camera_index"]}, "live")
                self.home_monitoring["error"] = None
            except Exception as exc:
                self.home_monitoring["error"] = str(exc)
                self.store.audit("home", "monitoring_retry", {"error": str(exc)})

    @staticmethod
    def describe(event):
        """事件 + 语音问询结论。结论是只读推导，不写回数据库。"""
        return {**event, "voice": voice_inquiry(event), "withdrawn": is_withdrawn(event), **withdraw_status(event)}

    def describe_all(self, **filters):
        return [self.describe(event) for event in self.store.events(**filters)]

    def event_detail(self, event_id):
        return self.describe(self.store.event(event_id))

    def signal_offline(self, path, source):
        result = self.modules["signal"].call("offline", path=str(path))
        run = uuid.uuid4().hex
        records = [self.events.create("accel", evidence, "replay", source, f"signal:{run}:{index}")
                   for index, evidence in enumerate(result["events"])]
        self.store.audit("signal", "offline_completed", {"windows": result["info"]["n_windows"], "events": len(records)})
        return {**result, "records": records, "mode": "replay"}

    def signal_results(self, result, stream_id):
        for alarm in result.get("alarms", []):
            self.events.create("accel", alarm, "live", "实时传感器", f"sensor:{stream_id}:{alarm['t_alarm']}",
                               at=datetime.fromtimestamp(alarm["wall_alarm"], timezone.utc).isoformat(timespec="milliseconds"))
        for evidence in result.get("events", []):
            event = self.events.create("accel", evidence, "live", "实时传感器", f"sensor:{stream_id}:{evidence['t_alarm']}",
                at=datetime.fromtimestamp(evidence.get("wall_alarm") or time.time(), timezone.utc).isoformat(timespec="milliseconds"))
            if "snippet" not in event["evidence"]:
                self.store.update_event(event["id"], lambda item, e=evidence: item["evidence"].update(e))

    def create_session(self, exercise, target, mode, action_scope="open"):
        with self.sessions_lock:
            active = [m for m in self.session_metadata.values() if m.get("active")]
            if active:
                raise ValueError("请先结束当前运动训练")
            inactive = [sid for sid, meta in self.session_metadata.items() if not meta["active"]]
            for old_sid in inactive[:-99]:
                self.session_metadata.pop(old_sid, None)
                self.session_commands.pop(old_sid, None)
                self.session_cancel.pop(old_sid, None)
                self.latest_poses.pop(old_sid, None)
                self.camera_shapes.pop(old_sid, None)
            result = self.modules["exercise"].call("create", exercise=exercise, target_reps=target, action_scope=action_scope)
            sid = result["session_id"]
            self.session_metadata[sid] = {"created_at": now(), "mode": mode, "active": True}
            self.session_commands[sid] = threading.RLock()
            self.session_cancel[sid] = threading.Event()
            return self.save_session(result)

    def save_session(self, result):
        sid = result["session_id"]
        meta = self.session_metadata[sid]
        meta["status"] = result["status"]
        payload = {**result, "created_at": meta["created_at"], "updated_at": now(), "mode": meta["mode"]}
        if result["status"] == "ended":
            payload["ended_at"] = now()
            meta["active"] = False
        return self.store.save_session(payload)

    def session_control(self, sid, action):
        if sid not in self.session_metadata:
            raise KeyError(sid)
        with self.session_commands[sid]:
            if not self.session_metadata[sid]["active"]:
                if action == "end":
                    return next(s for s in self.store.sessions() if s["session_id"] == sid)
                raise ValueError("训练已经结束")
            if action == "end":
                self.session_cancel[sid].set()
            result = self.save_session(self.modules["exercise"].call("control", session_id=sid, action=action))
            if action == "end":
                try:
                    trace = self.modules["exercise"].call("trace", session_id=sid)
                    directory = RUNTIME / "exercise-traces"
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / f"{sid}.json").write_text(json.dumps(trace), encoding="utf-8")
                except Exception:
                    log.exception("Could not retain exercise input diagnostics for %s", sid)
                finally:
                    try:
                        self.modules["exercise"].call("release", session_id=sid)
                    finally:
                        self.cameras.release(sid)
            return result

    def session_trace(self, sid):
        if not any(s["session_id"] == sid for s in self.store.sessions()):
            raise KeyError(sid)
        if sid in self.session_metadata:
            with self.session_commands[sid]:
                if self.session_metadata[sid]["active"]:
                    return self.modules["exercise"].call("trace", session_id=sid)
        path = RUNTIME / "exercise-traces" / f"{sid}.json"
        if not path.is_file():
            raise ValueError("这次训练未保留骨架诊断，请开始新的训练")
        return json.loads(path.read_text(encoding="utf-8"))

    def start_camera_session(self, exercise, target, index, handover=None):
        with self.sessions_lock:
            if any(meta["active"] for meta in self.session_metadata.values()):
                raise ValueError("请先结束当前运动训练")
            owner = "exercise-opening"
            if not handover:
                self.cameras.acquire(index, owner)
            try:
                self.modules["exercise"].call("pose_initialize")
                result = self.create_session(exercise, target, "live", action_scope="training")
                sid = result["session_id"]
                descriptor = self.handover_camera(handover or owner, sid, index)
                self.session_metadata[sid]["shared_camera"] = True
                self.session_metadata[sid]["last_viewer"] = time.monotonic()
                self.camera_shapes[sid] = descriptor["shape"]
                self.submit(self.camera_exercise, sid, descriptor)
                return {**result, "shared_camera": True, "frame_shape": descriptor["shape"]}
            finally:
                self.cameras.release(owner)

    def handover_camera(self, owner, sid, index):
        """把一路正在采样的画面从 ``owner`` 直接交给训练会话，不关设备、不丢帧。

        后台常驻在视频跌倒与运动检测之间轮换，两者用的是同一个摄像头；
        原来的 ``acquire/release`` 会拆机重开，画面要重新协商（约一秒黑屏），
        这里改为只替换占用者编号，像素管道保持不动。
        """
        previous = self.cameras.descriptor(owner)
        descriptor = self.cameras.acquire(previous["index"] if previous else index, sid)
        # Register the training session before dropping the previous owner.
        self.cameras.release(owner)
        return descriptor

    def camera_exercise(self, sid, descriptor):
        sequence = 0
        base = time.perf_counter()
        try:
            while not self.session_cancel[sid].is_set() and not self.stop_event.is_set():
                with self.session_commands[sid]:
                    if self.session_cancel[sid].is_set() or self.stop_event.is_set():
                        break
                    if self.session_metadata[sid].get("status") == "running":
                        result = self.modules["exercise"].call("camera_frame", session_id=sid,
                            descriptor=descriptor, base_stamp=base, last_sequence=sequence)
                        if not result.get("duplicate"):
                            sequence = result.pop("capture_sequence")
                            self.latest_poses[sid] = result.pop("keypoints")
                            self.save_session(result)
                if self.session_cancel[sid].wait(.005):
                    break
        except Exception as exc:
            self.interrupt_session(sid, exc)
        finally:
            # Release the worker's memory handle before the last capture owner.
            with self.session_commands[sid]:
                if self.modules["exercise"].state == "ready":
                    try:
                        self.modules["exercise"].call("camera_detach", session_id=sid)
                    except Exception:
                        pass
            self.cameras.release(sid)

    def check_sessions(self):
        active = {sid for sid, meta in self.session_metadata.items() if meta["active"] and meta["mode"] == "live"}
        if not active:
            return
        for item in self.store.sessions(mode="live"):
            meta = self.session_metadata.get(item["session_id"], {})
            if item["session_id"] in active and meta.get("shared_camera") and time.monotonic() - meta["last_viewer"] > 15:
                self.session_control(item["session_id"], "end")
                self.store.audit("exercise", "client_disconnected", {"session_id": item["session_id"]})
                continue
            if item["session_id"] in active and item["status"] == "running":
                stamp = datetime.fromisoformat(item.get("updated_at", item["created_at"])).timestamp()
                if time.time() - stamp > 10:
                    self.session_control(item["session_id"], "pause")
                    self.store.audit("exercise", "capture_timeout", {"session_id": item["session_id"]})

    def session_frames(self, sid, body):
        if sid not in self.session_metadata:
            raise KeyError(sid)
        with self.session_commands[sid]:
            result = self.modules["exercise"].call("frames", session_id=sid, body=body)
            self.latest_poses[sid] = body["frames"][-1]["keypoints"]
            return self.save_session(result)

    def session_image(self, sid, data, timestamp_ms, request_id):
        if sid not in self.session_metadata:
            raise KeyError(sid)
        with self.session_commands[sid]:
            result = self.modules["exercise"].call("image", session_id=sid, image=base64.b64encode(data).decode(),
                        timestamp_ms=timestamp_ms, request_id=request_id)
            self.latest_poses[sid] = result.pop("keypoints")
            self.save_session(result)
            return {**result, "keypoints": self.latest_poses[sid]}

    def replay(self, sid, exercise):
        record = json.loads((EXERCISE / "examples" / f"{exercise}_replay.json").read_text())
        try:
            for index in range(0, len(record["frames"]), 20):
                if self.session_cancel[sid].is_set() or self.stop_event.is_set():
                    return
                while self.session_metadata[sid]["active"]:
                    with self.session_commands[sid]:
                        snapshot = self.modules["exercise"].call("snapshot", session_id=sid)
                    if snapshot["status"] != "paused":
                        break
                    if self.session_cancel[sid].wait(.2):
                        return
                self.session_frames(sid, {"request_id": f"replay-{index}", "frames": record["frames"][index:index+20]})
                if self.session_cancel[sid].wait(.18):
                    return
            self.session_control(sid, "end")
        except Exception as exc:
            self.interrupt_session(sid, exc)

    def video_exercise(self, sid, path):
        try:
            self.modules["exercise"].call("pose_initialize")
            self.modules["exercise"].call("video_open", session_id=sid, path=str(path))
            while not self.session_cancel[sid].is_set() and not self.stop_event.is_set():
                with self.session_commands[sid]:
                    if self.session_cancel[sid].is_set():
                        return
                    if self.session_metadata[sid].get("status") == "paused":
                        paused = True
                    else:
                        paused = False
                        result = self.modules["exercise"].call("video_frame", session_id=sid)
                        if result.get("eof"):
                            break
                        self.latest_poses[sid] = result.pop("keypoints")
                        self.save_session(result)
                if paused and self.session_cancel[sid].wait(.2):
                    return
            if not self.session_cancel[sid].is_set():
                self.session_control(sid, "end")
        except Exception as exc:
            self.interrupt_session(sid, exc)
        finally:
            with self.session_commands[sid]:
                if self.modules["exercise"].state == "ready":
                    self.modules["exercise"].call("video_detach", session_id=sid)

    def interrupt_session(self, sid, error):
        log.exception("Exercise session %s interrupted", sid)
        with self.session_commands[sid]:
            sessions = self.store.sessions()
            current = next((s for s in sessions if s["session_id"] == sid), None)
            if current and current["status"] != "ended":
                current.update(status="interrupted", quality="module_error", error=str(error), ended_at=now())
                self.store.save_session(current)
            self.session_metadata[sid]["active"] = False
            self.session_cancel[sid].set()
            self.latest_poses.pop(sid, None)
            try:
                self.modules["exercise"].call("release", session_id=sid, timeout=10)
            except Exception:
                pass
            self.cameras.release(sid)

    def audio_analyze(self, data, task, mode, event_id=None):
        if not self.store.settings()["audio_consent"]:
            raise ValueError("需先在设置中获得长者的音频处理授权")
        result = self.modules["audio"].call("analyze", timeout=120, data=base64.b64encode(data).decode(), task=task)
        if not self.store.settings()["audio_consent"]:
            raise ValueError("音频授权已撤回，结果已丢弃")
        result.pop("transcript", None)
        if task == "reply":
            if not event_id:
                raise ValueError("语音回复需要关联一个事件")
            event = self.events.inquiry_reply(event_id, result)
            return {"result": result, "event": event}
        help_result = result.get("help_result", {}) if task == "monitor" else result if task == "help" else {}
        if help_result.get("help"):
            event = self.events.create("audio", help_result, mode, "麦克风" if mode == "live" else "音频回放")
            help_result["event_id"] = event["id"]
        stored = self.store.save_sound({"task": task, "result": result}, mode)
        return {"result": result, "id": stored["id"]}

    def deliver_webhooks(self):
        destination = self.store.settings().get("webhook_url")
        if not destination:
            return
        with self.store.connection() as conn:
            rows = conn.execute("SELECT * FROM notifications WHERE kind='webhook' AND status IN ('queued','retry') AND next_at<=? ORDER BY created_at LIMIT 1", (time.time(),)).fetchall()
        for row in rows:
            # The receiver gets the English interface text; the stored record keeps the original.
            sent = json.loads(row["payload"])
            sent.update({key: translate(sent[key]) for key in ("title", "message") if key in sent})
            try:
                with httpx.Client(timeout=5, follow_redirects=False) as client:
                    response = client.post(destination, json=sent, headers={"Idempotency-Key": row["id"]})
                    response.raise_for_status()
                state = "delivered"
            except Exception:
                state = "failed" if row["attempts"] >= 4 else "retry"
            with self.store.connection() as conn:
                payload = json.loads(row["payload"])
                payload["last_attempt_at"] = now()
                if state == "delivered":
                    payload["delivered_at"] = payload["last_attempt_at"]
                conn.execute("UPDATE notifications SET status=?,attempts=attempts+1,next_at=?,payload=? WHERE id=?", (
                    state, time.time() + min(300, 10 * 2 ** row["attempts"]), json.dumps(payload, ensure_ascii=False), row["id"]))

    @staticmethod
    def report_start(days=7):
        """报告窗口起点（本地日界，UTC）。删除范围与报告范围共用同一算法，避免错删。"""
        tz = ZoneInfo("Asia/Shanghai")
        today = datetime.now(tz).date()
        return datetime.combine(today - timedelta(days=days-1), datetime.min.time(), tzinfo=tz).astimezone(timezone.utc)

    def delete_records(self, event_ids=(), session_ids=()):
        """删除选中的事件与训练记录，连带通知与录像片段。"""
        result = self.records.events(event_ids)
        result.update(self.records.sessions(session_ids))
        self.store.audit("records", "deleted", {key: value for key, value in result.items() if value})
        return result

    def clear_report_scope(self, mode, days, events=True, sessions=True):
        """清空当前报告范围内的记录；范围与 ``report()`` 完全一致。"""
        result = self.records.scope(mode, self.report_start(days).isoformat(), events, sessions)
        self.store.audit("records", "scope_cleared", {"mode": mode, "days": days,
                                                      **{key: value for key, value in result.items() if value}})
        return result

    def report(self, days=7, mode="live"):
        tz = ZoneInfo("Asia/Shanghai")
        today = datetime.now(tz).date()
        start = self.report_start(days)
        events = self.describe_all(mode=mode, limit=100000, since=start.isoformat())
        sessions = self.store.sessions(mode=mode, since=start.isoformat(), limit=100000)
        coverage = self.store.coverage(start.isoformat()) if mode == "live" else []
        series = []
        for offset in range(days):
            date = today - timedelta(days=days-1-offset)
            same_day = lambda item: datetime.fromisoformat(item["created_at"]).astimezone(tz).date() == date
            daily_events = [e for e in events if same_day(e)]
            daily_sessions = [s for s in sessions if same_day(s)]
            series.append({"date": date.isoformat(), "events": len(daily_events), "reps": sum(s.get("count", 0) for s in daily_sessions),
                           "confirmed_falls": sum(e["outcome"] == "confirmed_fall" for e in daily_events)})
        unavailable = {}
        for row in coverage:
            end = datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else datetime.now(timezone.utc)
            begin = max(start, datetime.fromisoformat(row["started_at"]))
            if row["state"] != "active":
                unavailable[row["channel"]] = unavailable.get(row["channel"], 0) + max(0, (end - begin).total_seconds())
        completed = [s for s in sessions if s["status"] == "ended"]
        return {"generated_at": now(), "days": days, "mode": mode, "resident": self.store.settings()["resident"],
                "events": events, "sessions": sessions, "series": series, "coverage": coverage,
                "notifications": [n for n in self.store.notifications(since=start.isoformat(), limit=100000) if n.get("mode") == mode],
                "summary": {"confirmed_falls": sum(e["outcome"] == "confirmed_fall" for e in events),
                    "false_alarms": sum(e["outcome"] == "false_alarm" for e in events), "open_events": sum(e["state"] == "open" for e in events),
                    "verification_events": sum(e["level"] == "PENDING" for e in events),
                    "unresolved_outcomes": sum(e["outcome"] == "unresolved" for e in events),
                    "sessions": len(completed), "reps": sum(s.get("count", 0) for s in completed),
                    "participation_days": len({datetime.fromisoformat(s["created_at"]).astimezone(tz).date() for s in completed}),
                    "reps_by_exercise": {key: sum(s.get("count", 0) for s in completed if s["exercise"] == key) for key in EXERCISES},
                    "withdrawn": sum(bool(e.get("withdrawn")) for e in events),
                    "voice_answered": sum(e["voice"]["state"] == ANSWERED for e in events),
                    "voice_high_risk": sum(e["voice"]["state"] == HIGH_RISK for e in events),
                    "voice_not_asked": sum(e["voice"]["state"] == NOT_ASKED for e in events),
                    "coverage_unavailable_s": unavailable}}

    def close(self):
        self.stop_event.set()
        for cancel in self.session_cancel.values():
            cancel.set()
        if hasattr(self, "scheduler"):
            self.scheduler.join(timeout=8)
        self.pool.shutdown(wait=True, cancel_futures=True)
        for sid, meta in list(self.session_metadata.items()):
            if meta["active"]:
                try:
                    self.session_control(sid, "end")
                except Exception:
                    pass
        for module in self.modules.values():
            module.close()
        self.cameras.close()
        for channel in ("vision", "accel", "audio"):
            self.store.set_coverage(channel, "unavailable", "服务已停止")
