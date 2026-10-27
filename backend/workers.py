"""One original algorithm import namespace per process; JSON over local pipes."""
import base64
import contextlib
import importlib.util
import json
import os
import shutil
import sys
import traceback
import time
from datetime import date, datetime
from pathlib import Path

from backend.config import VIDEO, SIGNAL, EXERCISE, AUDIO, RUNTIME, prepare


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _map_cuda_pickles_to_cpu():
    """a4_weights.pkl was exported on a CUDA machine. Without CUDA, load its tensors onto the CPU.

    Returns whether the mapping was enabled.
    """
    import io
    import torch
    if torch.cuda.is_available():
        return False
    torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
    return True


class VideoWorker:
    def __init__(self):
        # The delivered engine expects systems/ beside standalone/. Mirror only
        # its integration wrapper byte-for-byte; all inference comes from originals.
        bridge = RUNTIME / "video_bridge"
        systems = bridge / "systems"
        standalone = bridge / "standalone"
        systems.mkdir(parents=True, exist_ok=True)
        standalone.mkdir(parents=True, exist_ok=True)
        for name in ("engine.py", "store.py", "a4_weights.pkl"):
            shutil.copy2(VIDEO / name, systems / name)
        for name in ("main.py", "a4_core.py", "yolov8n-pose.onnx", "pose_best.pt", "a4_weights.pkl"):
            target = standalone / name
            source = VIDEO / "standalone" / name
            if not target.exists() or target.stat().st_mtime_ns != source.stat().st_mtime_ns:
                shutil.copy2(source, target)
        self.module = load_file("homeease_original_video_engine", systems / "engine.py")
        import cv2
        cv2.setNumThreads(max(1, int(os.environ.get("HOMEEASE_VIDEO_CV_THREADS", "4"))))
        original_source = self.module.FrameSource

        class SharedSource(original_source):
            def __init__(self, spec):
                self.reader = None
                if spec.get("type") == "camera" and spec.get("shared_camera"):
                    from backend.camera import FrameReader
                    self.reader = FrameReader(spec["shared_camera"])
                    self.realtime = True
                    self.label = f"Camera {spec.get('index', 0)}"
                else:
                    super().__init__(spec)

            def read(self):
                if self.reader:
                    frame, _, _ = self.reader.read()
                    return True, frame
                return super().read()

            def release(self):
                if self.reader:
                    self.reader.close()
                else:
                    super().release()

        self.module.FrameSource = SharedSource
        cpu_only = _map_cuda_pickles_to_cpu()
        # 画面叠加交给工作台前端（检测框与骨架可各自开关），因此让原引擎输出干净画面；
        # 这只是给原引擎构造函数传显示参数，不涉及任何检测逻辑或阈值。
        # 「录像叠加」开关可随时用原引擎的 set_config 把框线烙回画面与录像。
        self.engine = self.module.DetectionEngine(show_skeleton=False)
        if not self.engine.models_ready:
            raise RuntimeError(self.engine._load_error)
        if cpu_only:
            # The temporal networks in the weights remember the CUDA device they were trained on;
            # without CUDA they run on the CPU. Weights and decisions are unchanged.
            import torch
            for base in self.engine._alg.bases:
                if getattr(base, "device", None) is not None and base.device.type == "cuda":
                    base.device = torch.device("cpu")
        self.detector_backend = "opencv"
        self.acceleration_error = None
        if os.environ.get("HOMEEASE_VIDEO_DNN", "auto") != "opencv":
            try:
                from backend.adapters.onnx import OnnxNetwork
                self.engine._det.net = OnnxNetwork(VIDEO / "standalone" / "yolov8n-pose.onnx", device="cuda")
                self.detector_backend = "onnxruntime-cuda-fp32"
            except Exception as exc:
                self.acceleration_error = str(exc)
                print(f"Video CUDA acceleration unavailable; original OpenCV backend: {exc}", file=sys.stderr)
        # All native libraries are loaded now. Reuse the library inventory while
        # retaining the original per-prediction limits and restoration semantics.
        from threadpoolctl import ThreadpoolController
        self.threadpools = ThreadpoolController()
        self.module._threadpool_limits = self.threadpools.limit
        import numpy as np
        warm_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.engine._det.detect_frame(warm_frame)
        self.engine._pose.extract(warm_frame, [0, 0, 640, 480])
        with self.module.a4_fast_context():
            self.engine._alg.predict(np.zeros((30, 17, 3), dtype=np.float32))
        # 画面叠加层用的只读缓存：按引擎帧号缓存当前画面尺寸，避免每次调用都解码 JPEG。
        self._pose_frames = -1
        self._pose_shape = (0, 0)

    def frame_shape(self):
        """当前画面尺寸（宽, 高）；只在引擎帧号变化时解码一次最新 JPEG。"""
        import cv2
        import numpy as np
        frames = int(getattr(self.engine, "_frames", 0))
        if frames != self._pose_frames:
            encoded = self.engine.latest_jpeg()
            if encoded:
                image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is not None:
                    self._pose_shape = (int(image.shape[1]), int(image.shape[0]))
            self._pose_frames = frames
        return self._pose_shape

    def pose(self):
        """只读快照：原引擎最后一帧的人体框与 17 个关键点（均归一化到 0–1）。

        供前端在自己的画布上叠加显示（检测框 / 骨架各自独立开关）。这里只读取原引擎
        已经算好的结果，不参与检测判定，也不改变送进模型的任何输入。
        """
        width, height = self.frame_shape()
        window = list(getattr(self.engine, "_kpts", ()) or ())
        latest = window[-1] if window else None
        points = [[round(float(x), 5), round(float(y), 5), round(float(score), 4)]
                  for x, y, score in latest] if latest is not None else []
        box = getattr(self.engine, "_last_box", None)
        return {"frames": int(getattr(self.engine, "_frames", 0)), "width": width, "height": height,
                "threshold": float(self.engine.threshold), "prob": round(float(self.engine._prob), 4),
                "person_visible": getattr(self.engine, "_last_det_n", 0) == 1,
                "box": [round(float(value), 5) for value in box] if box else None,
                "keypoints": points}

    def initialize(self):
        import cv2
        return {"model": "A4", "pipeline": "YOLOv8n + P1 + A4", "config": self.engine.config(),
                "opencv_threads": cv2.getNumThreads(), "detector_backend": self.detector_backend,
                "acceleration_error": self.acceleration_error}

    def start(self, source):
        if self.engine.is_running():
            raise ValueError("Detection is running. Please stop it first")
        return self.engine.start(source)

    def stop(self):
        result = self.engine.stop(timeout=8)
        return result

    def status(self):
        thread = self.engine._thread
        alive = bool(thread and thread.is_alive())
        return {**self.engine.status(), "records": self.engine.records()[:200],
                "person_visible": self.engine._last_det_n == 1, "alarm_at": self.engine._last_alarm,
                "processing": alive}

    def frame(self):
        return base64.b64encode(self.engine.latest_jpeg()).decode("ascii")

    def close(self):
        self.engine.stop(timeout=8)


class SignalWorker:
    def __init__(self):
        sys.path.insert(0, str(SIGNAL))
        from core.model import BestFallModel
        from core import detector, preprocess
        self.model = BestFallModel.load(str(SIGNAL / "model" / "best_model.pkl"))
        self.detector, self.preprocess = detector, preprocess
        self.stream = None

    def initialize(self):
        from core import config
        return {"model": self.model.name, "threshold": config.ALARM_THRESHOLD, "meta": self.model.meta}

    def offline(self, path):
        t, a = self.preprocess.read_acc_csv(path)
        if len(t) < 12:
            raise ValueError("The CSV does not have enough valid data. The original reader needs 5 columns: milliseconds, magnitude, ax, ay, az")
        events, info = self.detector.detect_offline(t, a, self.model)
        step = max(1, len(t) // 2000)
        return {"events": events, "info": info, "waveform": {"t": t[::step].tolist(), "ax": a[::step, 0].tolist(),
                "ay": a[::step, 1].tolist(), "az": a[::step, 2].tolist()}}

    def stream_start(self):
        if self.stream is not None:
            raise ValueError("The sensor stream is running")
        self.stream = self.detector.StreamingDetector(self.model)
        return {"state": "listening"}

    def samples(self, samples):
        if self.stream is None:
            raise ValueError("Please start the sensor stream first")
        events = []
        for sample in samples:
            events.extend(self.stream.push(*sample))
        alarms = self.stream.pending_alarms[:]
        self.stream.pending_alarms.clear()
        return {"events": events, "alarms": alarms, "prob": self.stream.last_prob, "windows": self.stream.n_eval,
                "recent": self.stream.recent[-120:]}

    def flush(self):
        if self.stream is None:
            return {"events": []}
        events = self.stream.flush()
        self.stream = None
        return {"events": events}

    def close(self):
        self.stream = None


class ExerciseWorker:
    def __init__(self):
        import torch
        from fastapi.testclient import TestClient
        sys.path.insert(0, str(EXERCISE))
        from homeease.api import create_app
        from homeease.engine import Predictor
        requested = os.environ.get("HOMEEASE_EXERCISE_DEVICE", "auto")
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
        predictor = Predictor(device=self.device)
        import numpy as np
        predictor.stream().update(np.zeros((12, 3), dtype=np.float32), 0)
        self.client = TestClient(create_app(predictor=predictor))
        self.client.__enter__()
        self.pose = None
        self.camera_readers = {}
        self.video_readers = {}
        self.diagnostics = {}

    def initialize(self):
        return {**self._request("get", "/health"), "device": self.device}

    def _request(self, method, path, body=None):
        response = self.client.request(method, path, json=body)
        if response.status_code >= 400:
            from fastapi import HTTPException
            raise HTTPException(response.status_code, response.json())
        return response.json() if response.content else {}

    def create(self, exercise, target_reps, action_scope="open"):
        from backend.exercise import ExerciseDiagnostics
        result = self._request("post", "/v1/sessions", {"exercise": exercise, "target_reps": target_reps,
            "action_scope": action_scope,
            "counting_policy": ("pose_cycle_v2" if exercise == "chair_sit_to_stand" else "pose_cycle_v1")
                if action_scope == "training" and exercise in {"chair_sit_to_stand", "standing_leg_raise"} else "original"})
        self.diagnostics[result["session_id"]] = ExerciseDiagnostics()
        return self.diagnostics[result["session_id"]].attach(result)

    def frames(self, session_id, body):
        result = self._request("post", f"/v1/sessions/{session_id}/frames", body)
        diagnostic = self.diagnostics[session_id]
        # A retry is accepted by the original API but must not inflate diagnostics.
        frames = [f for f in body["frames"] if not diagnostic.stamps or f["timestamp_ms"] > diagnostic.stamps[-1]]
        diagnostic.observe(frames)
        return diagnostic.attach(result)

    def snapshot(self, session_id):
        return self.diagnostics[session_id].attach(self._request("get", f"/v1/sessions/{session_id}"))

    def control(self, session_id, action):
        return self.diagnostics[session_id].attach(
            self._request("post", f"/v1/sessions/{session_id}/control", {"action": action}))

    def trace(self, session_id):
        diagnostic = self.diagnostics[session_id]
        return {"schema_name": "coco17", "coordinate_space": "camera_xy",
                "frames": list(diagnostic.recent_frames), "state": self.snapshot(session_id),
                "truncated": diagnostic.observed_frames > len(diagnostic.recent_frames)}

    def release(self, session_id):
        self.camera_detach(session_id)
        self.video_detach(session_id)
        result = self._request("delete", f"/v1/sessions/{session_id}")
        self.diagnostics.pop(session_id, None)
        return result

    def video_open(self, session_id, path):
        import cv2
        import math
        self.video_detach(session_id)
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not math.isfinite(fps) or not 1 <= fps <= 240:
            cap.release()
            raise ValueError("Could not open the exercise video or determine its frame rate")
        self.video_readers[session_id] = {"capture": cap, "fps": fps, "index": 0}
        return {"fps": fps}

    def video_frame(self, session_id):
        source = self.video_readers[session_id]
        ok, frame = source["capture"].read()
        if not ok:
            return {"eof": True}
        index = source["index"]
        result = self._pose_frame(session_id, frame, index * 1000 / source["fps"], f"video-{index}")
        source["index"] += 1
        return {**result, "frame_shape": list(frame.shape)}

    def video_detach(self, session_id):
        source = self.video_readers.pop(session_id, None)
        if source:
            source["capture"].release()
        return {"detached": True}

    def camera_detach(self, session_id):
        reader = self.camera_readers.pop(session_id, None)
        if reader:
            reader.close()
        return {"detached": True}

    def pose_initialize(self):
        if self.pose is None:
            import numpy as np
            from homeease.pose import PoseEstimator
            self.pose = PoseEstimator(device=0 if self.device.startswith("cuda") else self.device)
            # Ultralytics/CUDA warm-up must finish before a live session's clock.
            self.pose.estimate(np.zeros((480, 640, 3), dtype=np.uint8))
        return {"model": "yolo11n-pose.pt", "device": self.device}

    def image(self, session_id, image, timestamp_ms, request_id):
        import cv2
        import numpy as np
        self.pose_initialize()
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(image), np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Cannot read the image")
        return self._pose_frame(session_id, frame, timestamp_ms, request_id)

    def camera_frame(self, session_id, descriptor, base_stamp, last_sequence):
        from backend.camera import FrameReader
        reader = self.camera_readers.get(session_id)
        if reader is None:
            reader = self.camera_readers[session_id] = FrameReader(descriptor)
        frame, sequence, stamp = reader.read()
        if sequence == last_sequence or stamp < base_stamp:
            return {"duplicate": True}
        self.pose_initialize()
        result = self._pose_frame(session_id, frame, (stamp - base_stamp) * 1000, f"camera-{sequence}")
        return {**result, "capture_sequence": sequence, "frame_shape": list(frame.shape)}

    def _pose_frame(self, session_id, frame, timestamp_ms, request_id):
        started = time.perf_counter()
        points, person_id = self.pose.estimate(frame)
        pose_ms = (time.perf_counter() - started) * 1000
        result = self.frames(session_id, {"request_id": request_id, "frames": [{"timestamp_ms": timestamp_ms,
                              "person_id": person_id, "keypoints": points.tolist()}]})
        diagnostic = self.diagnostics[session_id]
        diagnostic.capture.update(pose_ms=round(pose_ms, 2),
            processing_ms=round((time.perf_counter() - started) * 1000, 2), device=self.device)
        return {**diagnostic.attach(result), "keypoints": points.tolist(), "person_id": person_id}

    def close(self):
        for sid in list(self.video_readers):
            self.video_detach(sid)
        for reader in self.camera_readers.values():
            reader.close()
        self.camera_readers.clear()
        self.diagnostics.clear()
        self.client.__exit__(None, None, None)


class AudioWorker:
    def __init__(self):
        import os
        os.environ["HOMEEASE_ROOT"] = str(AUDIO)
        sys.path.insert(0, str(AUDIO / "src"))
        from algorithm3 import Algorithm3
        # Strict Silero availability: never silently substitute another VAD.
        from homeease.ml.audio.vad import create
        self.algorithm = Algorithm3(vad="silero")
        self.algorithm._vad = create("silero")
        self.algorithm.vad_name = "silero"
        self.pipe = None
        self.listener = None
        self.inquiry_since = None

    def initialize(self):
        result = self.algorithm.warm_up()
        if result.get("sound_model_error") or result.get("asr_error"):
            raise RuntimeError(result.get("sound_model_error") or result.get("asr_error"))
        return result

    def analyze(self, data, task, text=None):
        if task == "reply_text":
            return self.algorithm.classify_reply(text=text)
        import numpy as np
        x = self.algorithm.decode(base64.b64decode(data))
        try:
            if len(x) > 16000 * 120:
                raise ValueError("Audio can be at most 120 seconds")
            if task == "monitor":
                sounds = self.algorithm.label_sounds(x)
                help_result = self.algorithm.detect_spoken_help(x)
                return {"sounds": sounds, "help_result": help_result, "duration_s": len(x) / 16000,
                        "raw_audio_stored": False}
            if task == "sounds":
                return self.algorithm.label_sounds(x)
            if task == "help":
                return self.algorithm.detect_spoken_help(x)
            if task == "reply":
                return self.algorithm.classify_reply(audio=x)
            raise ValueError("Unknown audio task")
        finally:
            x.fill(0)

    def outcome(self, record, level="PENDING", resolved=False, button_ok=False):
        return self.algorithm.inquiry_outcome(record, decision_level=level, resolved=resolved, button_ok=button_ok)

    def tag(self, markers):
        return self.algorithm.alarm_tag(markers)

    def live_start(self):
        from collections import deque
        from homeease.core.clock import SystemClock
        from homeease.ml.audio.monitor import SoundMonitor
        from homeease.ml.audio.pipeline import Function3Pipeline
        from homeease.ml.audio.help_listener import HelpListener
        if self.pipe is not None:
            raise ValueError("Microphone listening is running")
        self.clock = SystemClock()
        self.calls = deque(maxlen=32)
        self.inquiry_since = None
        algorithm = self.algorithm

        class LockedASR:
            def transcribe(self, *args, **kwargs):
                with algorithm._lock:
                    return algorithm._get_asr().transcribe(*args, **kwargs)

        self.listener = HelpListener(algorithm.cfg, self.clock, self.calls.append, consent=lambda: self.pipe is not None,
                                     asr=LockedASR(), inquiry_since=lambda: self.inquiry_since)
        self.monitor = SoundMonitor(algorithm.cfg.audio.sound_monitor, algorithm._get_classifier(), self.clock, store=None)
        self.pipe = Function3Pipeline(algorithm.cfg.audio, self.clock, self.monitor, voice_sink=self.listener.submit, vad=algorithm._get_vad())
        self.listener.start()
        return {"state": "listening", "sample_rate": 16000}

    def live_chunk(self, pcm, timestamp_ms, inquiry_active=False):
        import numpy as np
        from dataclasses import asdict
        if self.pipe is None:
            raise ValueError("Listening has not started")
        if inquiry_active and self.inquiry_since is None:
            self.inquiry_since = self.clock.mono_ms()
        elif not inquiry_active:
            self.inquiry_since = None
        x = np.frombuffer(base64.b64decode(pcm), dtype="<f4").copy()
        try:
            if not 1 <= len(x) <= 32000 or not np.isfinite(x).all():
                raise ValueError("Invalid audio data block")
            if not hasattr(self, "stream_base") or self.stream_base is None:
                self.stream_base = self.clock.mono_ms() - float(timestamp_ms) - len(x) / 16
            self.pipe.process_chunk(x, self.stream_base + float(timestamp_ms))
            calls = list(self.calls)
            self.calls.clear()
            return {"events": [asdict(e) for e in self.monitor.recent[-30:]], "calls": calls,
                    "pipeline": self.pipe.status(), "listener": self.listener.status(), "raw_audio_stored": False}
        finally:
            x.fill(0)

    def live_stop(self):
        from dataclasses import asdict
        pipe, self.pipe = self.pipe, None
        if pipe:
            pipe.close(timeout=5)
        if self.listener:
            self.listener.stop()
            self.listener = None
        self.stream_base = None
        self.inquiry_since = None
        calls = list(getattr(self, "calls", []))
        if hasattr(self, "calls"):
            self.calls.clear()
        return {"state": "stopped", "calls": calls,
                "events": [asdict(e) for e in self.monitor.recent[-30:]] if pipe else []}

    def close(self):
        self.live_stop()


def protocol_default(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"Unsupported worker result type: {type(value).__name__}")


def main():
    prepare()
    protocol = sys.stdout
    sys.stdout = sys.stderr
    instance = None
    classes = {"video": VideoWorker, "signal": SignalWorker, "exercise": ExerciseWorker, "audio": AudioWorker}
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if instance is None:
                    instance = classes[sys.argv[1]]()
                method = request["method"]
                if method.startswith("_") or method == "close":
                    raise ValueError("Unknown call")
                result = getattr(instance, method)(**request.get("params", {}))
                response = {"ok": True, "result": result, "worker_pid": os.getpid()}
                encoded = json.dumps(response, ensure_ascii=True, allow_nan=False, default=protocol_default)
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                response = {"ok": False, "fatal": instance is None, "error": str(getattr(exc, "detail", exc)),
                            "status": getattr(exc, "status_code", 422), "worker_pid": os.getpid()}
                encoded = json.dumps(response, ensure_ascii=True, allow_nan=False)
            protocol.write(encoded + "\n")
            protocol.flush()
    finally:
        if instance:
            instance.close()


if __name__ == "__main__":
    main()
