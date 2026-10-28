"""Input diagnostics for the original exercise API; never changes its decisions."""
from collections import Counter, deque


REQUIRED_JOINTS = {5: "left_shoulder", 6: "right_shoulder", 11: "left_hip", 12: "right_hip",
                   13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"}
TRACE_WINDOW_MS = 180000
TRACE_MAX_FRAMES = 1800


class ExerciseDiagnostics:
    def __init__(self):
        self.observed_frames = 0
        self.visible_frames = 0
        self.missing = Counter()
        self.stamps = deque(maxlen=60)
        self.recent_frames = deque(maxlen=TRACE_MAX_FRAMES)
        self.max_gap_ms = 0.0
        self.capture = {}

    def observe(self, frames):
        for frame in frames:
            stamp = frame["timestamp_ms"]
            gap = stamp - self.stamps[-1] if self.stamps else None
            if gap is not None:
                self.max_gap_ms = max(self.max_gap_ms, gap)
            self.stamps.append(stamp)
            self.recent_frames.append({"timestamp_ms": stamp, "keypoints": frame["keypoints"],
                                       "person_id": frame.get("person_id", "primary")})
            while self.recent_frames[0]["timestamp_ms"] < stamp - TRACE_WINDOW_MS:
                self.recent_frames.popleft()
            points = frame["keypoints"]
            missing = [name for i, name in REQUIRED_JOINTS.items() if points[i][2] < .3]
            self.observed_frames += 1
            self.visible_frames += not missing
            self.missing.update(missing)
            span = self.stamps[-1] - self.stamps[0]
            self.capture.update(timestamp_ms=stamp, gap_ms=gap, missing_joints=missing,
                joint_confidences={name: float(points[i][2]) for i, name in REQUIRED_JOINTS.items()},
                observation_fps=round((len(self.stamps) - 1) * 1000 / span, 1) if span else 0)

    def attach(self, result):
        interruptions = result.get("interruptions", [])
        latest = interruptions[-1] if interruptions else None
        stamp = result.get("last_timestamp_ms")
        uninterrupted = result.get("elapsed_s", 0)
        if latest and stamp is not None and latest["timestamp_ms"] is not None:
            uninterrupted = max(0, (stamp - latest["timestamp_ms"]) / 1000)
        if result["quality"] != "good" or result["status"] != "running":
            uninterrupted = 0
        return {**result, "capture_diagnostics": dict(self.capture), "counting_diagnostics": {
            "observed_frames": self.observed_frames, "visible_pose_frames": self.visible_frames,
            "visible_pose_ratio": self.visible_frames / self.observed_frames if self.observed_frames else 0,
            "missing_joint_frames": dict(self.missing), "max_gap_ms": self.max_gap_ms,
            "interruption_counts": dict(Counter(item["reason"] for item in interruptions)),
            "last_interruption": latest, "uninterrupted_s": uninterrupted}}
