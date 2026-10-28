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

from backend.adapters.process import AlgorithmProcess, ModuleError
from backend.camera import CameraHub
from backend.config import EXERCISE, SIGNAL, VIDEO, RUNTIME, EXERCISES, ALGORITHM_KEYS
from backend.events import Events
from backend.inquiry import NOT_ASKED, ANSWERED, HIGH_RISK, voice_inquiry
from backend.events import is_withdrawn, withdraw_status
from backend.records import Records
from backend.storage import Store, now, ALGORITHM_DEFAULTS, PLAN_DEFAULTS

log = logging.getLogger(__name__)

EXERCISE_KEYS = tuple(EXERCISES)
# Marker for a camera that cannot be opened (not plugged in, or held by other software): tells a device problem apart from a passing failure.
FAILED = "Camera unavailable"


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
            raise ValueError("The CSV contains unreadable rows")
        if len(values) not in {4, 5} or not all(math.isfinite(v) for v in values):
            raise ValueError("The CSV needs 4 or 5 columns of finite numbers")
        numeric.append(values)
    if len(numeric) < 12:
        raise ValueError("At least 12 valid acceleration samples are needed")
    widths = {len(r) for r in numeric}
    if len(widths) != 1:
        raise ValueError("The CSV rows have different numbers of columns")
    if len(numeric[0]) == 4:
        if not header or header[0] not in {"t", "t_s", "t_ms"}:
            raise ValueError("A 4-column CSV needs the header t,ax,ay,az (seconds) or t_ms,ax,ay,az (milliseconds); acceleration in g")
        factor = 1 if header[0] == "t_ms" else 1000
        numeric = [[r[0] * factor, math.sqrt(sum(v * v for v in r[1:])), *r[1:]] for r in numeric]
    if any(b[0] <= a[0] for a, b in zip(numeric, numeric[1:])):
        raise ValueError("Sample timestamps must strictly increase")
    output = io.StringIO(newline="")
    csv.writer(output).writerows(numeric)
    return output.getvalue().encode("utf-8")


def _camera_partners(sid):
    """Other sessions sharing this session's physical camera; they must release it before exclusive use."""
    if sid == "video":
        return {"exercise-opening"}
    return {"video", "exercise-opening"}


def _launcher():
    """Interpreter used for a restart: on Windows prefer pythonw.exe, so no console window pops up."""
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
        self.audio_quality = {"state": "active", "reason": "Microphone input is valid"}
        self.jobs = set()
        self.jobs_lock = threading.RLock()
        # Background algorithm switches: only the switch states and scheduling results are kept; they take no part in algorithm decisions.
        self.algorithms_lock = threading.RLock()
        self.algorithms = {key: item["enabled"] for key, item in self.store.algorithms().items()}
        self.remote_access = False
        # The two channels that run automatically in always-on mode. ``stop`` points to each channel's shutdown function,
        # so switching off only calls it and the routes need not know which mode is running.
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
            self.store.set_coverage(channel, "unavailable", "No live input connected yet")
        for old in self.store.sessions():
            if old["status"] in {"running", "paused"}:
                old.update(status="interrupted", quality="service_restarted", ended_at=now())
                self.store.save_session(old)
        # Older versions only had the "home monitoring" switch: after an upgrade it is aligned with "video fall detection",
        # so existing monitoring keeps running instead of being switched off silently by the new switch.
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

    # ------------------------------------------------------------------ background algorithm switches
    def algorithm_channel(self, key):
        """Read one switch; an unknown key returns None and the route answers 404."""
        with self.algorithms_lock:
            return self.algorithms.get(key)

    def set_algorithms(self, changes):
        """Change only the submitted switches; they take effect immediately (no need to wait for the next scheduling round).

        Switching on always-on mode only enables the mode itself; video and acceleration still follow their own switches.
        Switching it off stops every channel it started, but the switches are kept for the next time it is switched on.
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
                self.stop_vision("Video fall detection off", switch=True)
            if "accel" in changes and not changes["accel"]:
                self.stop_session_channel(self.sensor_session, "Accelerometer fall detection was switched off")
            for key, value in changes.items():
                if key in EXERCISE_KEYS and not value:
                    self.stop_session_channel(self.exercise_session, f"{EXERCISES[key]} detection off")
            if changes.get("resident"):
                self.resident["state"] = "starting"
                self.resident["error"] = None
                self.store.audit("resident", "enabled", {"remote_access": self.remote_access})
            elif "resident" in changes and not changes["resident"]:
                self.stop_vision("Always on in background off", switch=True)
                self.stop_session_channel(self.sensor_session, "Always on in background off")
                for key in EXERCISE_KEYS:
                    self.stop_session_channel(self.exercise_session, "Always on in background off")
                self.resident.update(state="off", error=None)
                self.store.audit("resident", "disabled", {})
            self.sync_home_monitoring()
            self.ensure_algorithms()
        except Exception:
            log.exception("Could not apply algorithm switches %s", changes)
        return self.algorithm_status()

    def sync_home_monitoring(self):
        """Always-on video and "home monitoring" are the same video fall monitoring; keep both in step.

        The pause button of the home page and the stop button of the workbench both go through here, so the switch never
        shows "on" while monitoring is actually paused, or the other way round.
        """
        with self.algorithms_lock:
            vision = self.algorithms.get("vision", True)
            resident = self.algorithms.get("resident", False)
        desired = bool(vision and (resident or self.home_monitoring["enabled"]))
        if desired != bool(self.home_monitoring["enabled"]):
            self.set_home_monitoring(desired, self.home_monitoring.get("camera_index", 0))
        return dict(self.home_monitoring)

    def stop_vision(self, reason, switch=False):
        """Stop video detection.

        ``switch=False`` is for internal hand-overs such as background rotation: only the running state stops, and the
        "video fall detection" switch must not be turned off with it; otherwise nobody would take the camera back afterwards.
        Only a caregiver switching it off passes ``switch=True``.
        """
        if self.video.get("state") != "running" and not self.home_monitoring["enabled"]:
            return
        # Note that set_home_monitoring(False) does not turn the vision switch off;
        # turning the switch off is written to the database by the caller (set_algorithms).
        self.set_home_monitoring(False, self.home_monitoring.get("camera_index", 0))
        self.store.audit("video", "stopped_by_switch", {"reason": reason, "switch": switch})

    @staticmethod
    def stop_session_channel(state, reason):
        """Switch off one background channel: end its session if there is one, otherwise just clear the "on" marker.

        The state is marked inactive first, so scheduling does not restart it within the same second; the actual
        shutdown (ending the session, releasing the camera) is left to the channel's own ``stop``.
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
        """Scheduling: hand every channel that is switched on but not running to the original algorithm.

        Every start goes through the existing video_start / start_camera_session, so input source, sampling
        and decisions are exactly those of the original algorithm; this only decides whether and when to start.
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
            self.resident.update(error=None)  # one more try: the device may have just been plugged back in
        # When the local state can already be read, only run the self-check; do not start the process again.
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
                return  # the camera is held by an exercise session; take over after the session ends
            if time.monotonic() < self.vision_retry_at:
                return
            if self.resident.get("error") == FAILED:
                self.resident.update(error=None)  # clear the lost marker first so the next round really tries again
            # Reuse the existing monitoring start path: same input, same rate-limited retries and error reporting.
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
        """The acceleration channel has no local capture device: in always-on mode only confirm who provides the data.

        Real three-axis data comes from the phone page /sensor (the browser pushes acceleration to /ws/sensor),
        or from a CSV replay. No data is faked here; the state and the reason are reported to Settings as they are.
        """
        state = self.sensor_session
        wanted = bool(switches.get("resident") and switches.get("accel", True))
        if state.get("active") and not wanted:
            self.stop_session_channel(state, "Accelerometer fall detection was switched off")
            return
        connected = self.sensor["state"] == "connected"
        with self.exercises_lock:
            state["active"] = bool(wanted and connected)
            state["sid"] = "sensor" if state["active"] else None
            state["error"] = None if not wanted or connected else "This computer has no three-axis sensor: open /sensor on the phone and keep that page in the foreground; samples stream here in real time"
        if state["active"]:
            state["stop"] = self._stop_sensor_channel
        else:
            state["stop"] = lambda reason: None

    def _stop_sensor_channel(self, reason):
        """Switching acceleration detection off does not disconnect a connected device; it only stops counting it as the background source."""
        with self.exercises_lock:
            self.sensor_session.update(active=False, sid=None, stop=lambda value: None)
        self.store.audit("resident", "accel_stopped", {"reason": reason})

    def ensure_exercises(self, switches):
        if self.exercise_session.get("error") == FAILED:
            return  # no usable camera here; background rotation would only open and close it repeatedly, wait for the device
        state = self.exercise_session
        wanted = [key for key in EXERCISE_KEYS if switches.get(key, True)]
        if not switches.get("resident"):
            wanted = []
        if state.get("active"):
            if state.get("key") in wanted:
                return
            self.stop_session_channel(state, "Detection for this movement was switched off")
            return
        if wanted and state.get("error") and (state.get("retry_at") or 0) > time.monotonic():
            return
        if not wanted or self.modules["exercise"].state != "ready":
            return
        # Exercise detection only takes over the stream the background opened itself. If a caregiver is watching the live
        # image on the monitoring workbench, the camera stays there and is taken over after video stops, instead of grabbing a watched image.
        handover = None
        if self.resident.get("error") == FAILED:
            return  # the camera cannot be opened; neither video nor exercise should retry every round
        if self.video.get("state") == "running" and self.video_mode == "live" and switches.get("vision", True):
            busy = self.cameras.busy("video")
            live = {sid for sid, meta in self.session_metadata.items() if meta.get("active")}
            if busy & live:
                return  # a person is exercising from the workbench; do not replace that session with background rotation
            if not busy:
                handover = "video"  # only background video holds the camera, so it can be handed over directly
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
            # The image now belongs to the exercise session and the video channel gives up the camera; the next round takes it back after the session.
            # keep_camera: the capture thread is in use by the exercise session right now and must not be closed.
            with self.video_lock:
                self.stop_vision("Camera handed over to background movement detection")

    def _next_exercise(self, wanted):
        """Rotate in plan order; a movement already completed this round makes way for the next, so the older adult does not wait."""
        with self.exercises_lock:
            last = self.exercise_session.get("last_key")
        plan = self.store.exercise_plan()
        order = [item["exercise"] for item in plan["items"] if item["enabled"] and item["exercise"] in wanted]
        order = order or list(wanted)
        if last in order:
            order = order[order.index(last) + 1:] + order[:order.index(last) + 1]
        return order[0]

    def clear_stale_sessions(self):
        """Clean up old sessions still marked running without an active session.

        They are left over from an earlier crash or a scheduling race: they have no in-memory state,
        never end by themselves, and would block "start a new session" forever.
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
        # End the session first, then clear the channel marker: the other way round, a session that failed to end
        # would be forgotten while still running and would hold the camera indefinitely.
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
        """Complete state for Settings: switches, what actually runs in the background now, and why something cannot run."""
        with self.algorithms_lock:
            switches = dict(self.algorithms)
        vision_on = self.video.get("state") == "running" and self.video_mode == "live"
        sensor = self.sensor_session
        exercise = self.exercise_session
        channels = [
            {"key": "vision", "type": "algorithm", "active": vision_on,
             "detail": self.video.get("error") or ("Live video detection running" if vision_on else "Not running")},
            {"key": "accel", "type": "algorithm",
             "active": bool(sensor.get("active")) or self.sensor["state"] == "connected",
             "detail": sensor.get("error") or ("Background capture running" if sensor.get("active") else
                                               "Phone sensor connected" if self.sensor["state"] == "connected" else "Not running")},
        ]
        current = exercise.get("key") if exercise.get("active") else None
        for key in EXERCISE_KEYS:
            running = current == key
            channels.append({"key": key, "type": "exercise", "active": running,
                             "detail": "Recognising movements with always-on" if running else
                                       "Ready to recognise" if switches.get("resident") and switches.get(key) else "Not running"})
        reasons = self.algorithm_blockers(switches)
        return {"switches": switches, "channels": channels, "resident": dict(self.resident),
                "resident_ready": self.resident_available(), "blockers": reasons,
                "remote_access": self.remote_access}

    def resident_available(self):
        return _script().is_file()

    def sensors_connected(self):
        """The acceleration channel's data comes from a three-axis device (the phone page /sensor); this machine has no second source."""
        return self.sensor["state"] == "connected"

    def plan_progress(self, plan=None):
        """Today's completion per movement; completion uses the existing session records (status=ended, mode=live)."""
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
        """"Not today" on the older adult's page: only record that today is skipped, without changing the plan.

        The plan itself is not moved to tomorrow: it is the caregiver's daily schedule, and only today's share is
        deferred. Once the skip date is stored, today's card disappears from the older adult's page and returns tomorrow.
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
            blockers.append("Background mode is off, so video, accelerometer and movement detection do not start by themselves")
            return blockers
        if not self.resident_available():
            blockers.append("The start script is missing, so always-on cannot recover by itself")
        if self.resident.get("error") and self.resident["error"] != FAILED:
            blockers.append(self.resident["error"])
        if self.resident.get("error") and self.resident["error"] == FAILED:
            blockers.append("The camera on this device is unavailable; it recovers once it is plugged in or freed")
        elif switches.get("vision") and self.modules["video"].state != "ready":
            blockers.append("The video algorithm is not ready yet; it takes over once available")
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
        """Person box and 17 keypoints of the current frame (normalized to 0-1) for the frontend overlay.

        The workbench frontend draws the detection box and skeleton on its own canvas, each with its own switch; this only
        reads results the original engine already computed, takes no part in decisions and exposes no detection parameter.
        """
        if self.modules["video"].state != "ready":
            raise ModuleError("The video module is not ready yet", 503)
        if self.video.get("state") not in {"running", "idle", "finished"}:
            raise ModuleError("Video detection is not running", 409)
        return self.modules["video"].call("pose", timeout=10)

    def video_start(self, source, mode):
        with self.video_lock:
            if self.home_monitoring["enabled"] and mode != "live":
                raise ValueError("Pause home monitoring before playing a video sample")
            if self.video.get("state") == "running":
                raise ValueError("Video detection is running")
            self.video_mode = mode
            self.video_run = uuid.uuid4().hex
            self.video_finalized = False
            self.video_event_keys = {}
            if source.get("type") == "camera":
                # Claim the camera before starting it: when acquire returns the capture thread already delivers frames, so a
                # handover can reuse it, and a failed video module start does not leave an unowned camera behind.
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
                # keep_camera: this stream is about to be handed to background exercise detection; do not close the device yet.
                self.cameras.release("video", accept=[keep_camera] if keep_camera else ())
            self.video_finalized = True
            self.video_run = None
            self.poll_video(force=True)
            self.store.set_coverage("vision", "unavailable", "Live video stopped")
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
                    Events.timeline(event, "evidence_saved", "The original video engine saved the event clip")
                self.store.update_event(eid, add_clip)

    def check_coverage(self):
        if self.video_mode == "live" and self.video.get("state") == "running":
            fresh = time.time() - self.video_last_frame_at < 8
            visible = self.video.get("person_visible", False)
            self.store.set_coverage("vision", "active" if fresh and visible else "reduced",
                "Live image is valid" if fresh and visible else ("No person detected in the image" if fresh else "Video data timed out"))
        else:
            self.store.set_coverage("vision", "unavailable", "Replaying a sample" if self.video_mode == "replay" and self.video.get("state") == "running" else "Live video is not on")
        if self.sensor["state"] == "connected":
            fresh = time.time() - (self.sensor.get("last_received") or 0) < 5
            self.store.set_coverage("accel", "active" if fresh else "reduced", "Live sensor data received" if fresh else "Sensor data timed out")
        else:
            self.store.set_coverage("accel", "unavailable", "Sensor not connected")
        consent = self.store.settings()["audio_consent"]
        fresh_audio = consent and self.audio_last_received and time.time() - self.audio_last_received < 20
        self.store.set_coverage("audio", self.audio_quality["state"] if fresh_audio else "unavailable",
            self.audio_quality["reason"] if fresh_audio else ("Audio consent is not on" if not consent else "No continuous microphone input connected"))

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
        """Start or stop "home monitoring". It is the same switch as "video fall detection" in Settings."""
        if not enabled:
            # Stopping only writes the running state and does not switch vision off: handing the camera to exercise detection
            # is an internal hand-over, and switching it off would leave nobody to restart video after the session.
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
        # "Home monitoring" and "video fall detection" in Settings are the same thing; keep the switches in step.
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
        """Event plus its voice-inquiry conclusion. The conclusion is derived read-only and not written back."""
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
            self.events.create("accel", alarm, "live", "Live sensor", f"sensor:{stream_id}:{alarm['t_alarm']}",
                               at=datetime.fromtimestamp(alarm["wall_alarm"], timezone.utc).isoformat(timespec="milliseconds"))
        for evidence in result.get("events", []):
            event = self.events.create("accel", evidence, "live", "Live sensor", f"sensor:{stream_id}:{evidence['t_alarm']}",
                at=datetime.fromtimestamp(evidence.get("wall_alarm") or time.time(), timezone.utc).isoformat(timespec="milliseconds"))
            if "snippet" not in event["evidence"]:
                self.store.update_event(event["id"], lambda item, e=evidence: item["evidence"].update(e))

    def create_session(self, exercise, target, mode, action_scope="open"):
        with self.sessions_lock:
            active = [m for m in self.session_metadata.values() if m.get("active")]
            if active:
                raise ValueError("Please end the current exercise session first")
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
                raise ValueError("The session has already ended")
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
            raise ValueError("No skeleton diagnostics were retained for this session. Please start a new session.")
        return json.loads(path.read_text(encoding="utf-8"))

    def start_camera_session(self, exercise, target, index, handover=None):
        with self.sessions_lock:
            if any(meta["active"] for meta in self.session_metadata.values()):
                raise ValueError("Please end the current exercise session first")
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
        """Hand a sampling stream from ``owner`` directly to an exercise session without closing the device or dropping frames.

        Always-on mode alternates between video fall detection and exercise detection on the same camera;
        the plain ``acquire/release`` would close and reopen it and renegotiate the image (about a second of black),
        so only the owner id is replaced and the pixel pipeline keeps running.
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
            raise ValueError("The older adult's audio-processing consent must be recorded in Settings first")
        result = self.modules["audio"].call("analyze", timeout=120, data=base64.b64encode(data).decode(), task=task)
        if not self.store.settings()["audio_consent"]:
            raise ValueError("Audio consent was withdrawn; the result was discarded")
        result.pop("transcript", None)
        if task == "reply":
            if not event_id:
                raise ValueError("A voice reply must be linked to an event")
            event = self.events.inquiry_reply(event_id, result)
            return {"result": result, "event": event}
        help_result = result.get("help_result", {}) if task == "monitor" else result if task == "help" else {}
        if help_result.get("help"):
            event = self.events.create("audio", help_result, mode, "Microphone" if mode == "live" else "Audio replay")
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
            sent = json.loads(row["payload"])
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
        """Start of the report window (local day boundary, in UTC). Deletion and reports share this function, so nothing else is deleted."""
        tz = ZoneInfo("Asia/Shanghai")
        today = datetime.now(tz).date()
        return datetime.combine(today - timedelta(days=days-1), datetime.min.time(), tzinfo=tz).astimezone(timezone.utc)

    def delete_records(self, event_ids=(), session_ids=()):
        """Delete the selected events and exercise sessions with their notifications and video clips."""
        result = self.records.events(event_ids)
        result.update(self.records.sessions(session_ids))
        self.store.audit("records", "deleted", {key: value for key, value in result.items() if value})
        return result

    def clear_report_scope(self, mode, days, events=True, sessions=True):
        """Clear the records inside the current report range; the range is exactly that of ``report()``."""
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
            self.store.set_coverage(channel, "unavailable", "Service stopped")
