"""Replay the held-out skeletons through the SDK, the HTTP API and a real video file."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from homeease.engine import ExerciseSession, Predictor
from homeease.exercises import EXERCISES
from homeease.pose import PoseEstimator

from .common import ROOT
from .data import project
from .metrics import replay


def coco_points(pose):
    result = np.zeros((17, 3), dtype=np.float32)
    result[5:] = pose
    result[:, :2] = result[:, :2] * 3 + np.array([960, 720])
    return result.tolist()


def demo(url=None):
    records = json.loads((ROOT / "data/processed/records.json").read_text())
    skeletons = np.load(ROOT / "data/processed/skeletons.npz")
    predictor = Predictor()
    reports = []
    example_dir = ROOT / "examples"
    example_dir.mkdir(exist_ok=True)
    for exercise, config in EXERCISES.items():
        source = replay(skeletons, records, "test", config["movement"])[0]
        poses = project(source["skel"])
        frames = [{"timestamp_ms": round(i / 30 * 1000, 6), "person_id": "primary", "keypoints": coco_points(p)} for i, p in enumerate(poses)]
        stream = ExerciseSession(predictor, exercise=exercise, target_reps=source["expected_count"])
        for frame in frames:
            stream.update(frame["timestamp_ms"], frame["keypoints"], frame["person_id"])
        result = stream.control("end")
        result["reference_count"] = source["expected_count"]
        result["source"] = source["id"]
        if url:
            import requests
            with requests.Session() as client:
                response = client.post(f"{url}/v1/sessions", json={"exercise": exercise, "target_reps": source["expected_count"]})
                response.raise_for_status()
                sid = response.json()["session_id"]
                for start in range(0, len(frames), 100):
                    response = client.post(f"{url}/v1/sessions/{sid}/frames", json={"request_id": f"{exercise}-{start}", "frames": frames[start:start + 100]})
                    response.raise_for_status()
                final = client.post(f"{url}/v1/sessions/{sid}/control", json={"action": "end"})
                final.raise_for_status()
                assert final.json()["count"] == result["count"], "HTTP and SDK count differ"
                result["http_count"] = final.json()["count"]
                client.delete(f"{url}/v1/sessions/{sid}").raise_for_status()
        (example_dir / f"{exercise}_frames.json").write_text(json.dumps({"request_id": "example-001", "frames": frames[:100]}, indent=2))
        (example_dir / f"{exercise}_replay.json").write_text(json.dumps({"exercise": exercise, "reference_count": source["expected_count"], "frames": frames}))
        reports.append(result)
        print(f"{exercise}: expected={source['expected_count']} predicted={result['count']}", flush=True)
    (ROOT / "reports/replay_sessions.json").write_text(json.dumps(reports, indent=2))


def video(source="0", target=10, exercise="chair_sit_to_stand", device=None, display=True, output="reports/video_session.json"):
    import cv2
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    webcam = source.isdecimal()
    capture = cv2.VideoCapture(int(source) if webcam else source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    pose = PoseEstimator(0 if device.startswith("cuda") else "cpu")
    session = ExerciseSession(Predictor(device=device), target_reps=target, exercise=exercise)
    started, index = time.monotonic(), 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            points, identity = pose.estimate(frame)
            timestamp = (time.monotonic() - started) * 1000 if webcam else index / fps * 1000
            index += 1
            if session.status == "running":
                state = session.update(timestamp, points, identity)
            else:
                state = session.snapshot()
            if display:
                cv2.putText(frame, f"{state['count']}/{target}  {state['phase']}  {state['quality']}",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 200, 80), 2)
                cv2.imshow("HomeEase exercise", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("p"):
                    session.control("pause" if session.status == "running" else "resume")
    finally:
        capture.release()
        cv2.destroyAllWindows()
    result = session.control("end")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def smoke():
    import cv2
    estimator = PoseEstimator(0 if torch.cuda.is_available() else "cpu")
    manifest = json.loads((ROOT / "data/coco/manifest.json").read_text())
    required = [5, 6, 11, 12, 13, 14, 15, 16]
    for entry in manifest:
        name = entry["url"].rsplit("/", 1)[-1]
        image = cv2.imread(str(ROOT / "data/coco" / name))
        points, person = estimator.estimate(image)
        if person == "primary" and points[required, 2].min() >= .3:
            break
    else:
        raise RuntimeError("No usable single-person image in the saved sample")
    h, w = image.shape[:2]
    path = ROOT / "reports/smoke_static_person.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Video encoder unavailable")
    for _ in range(60):
        writer.write(image)
    writer.release()
    video(str(path), display=False, output=str(ROOT / "reports/smoke_video_session.json"))
    result = json.loads((ROOT / "reports/smoke_video_session.json").read_text())
    result["test_scope"] = "real_pose_cnn_on_repeated_static_COCO_image_not_exercise_accuracy"
    result["image_source"] = entry
    assert result["count"] == 0, "Static image produced a spurious complete cycle"
    (ROOT / "reports/smoke_video_session.json").write_text(json.dumps(result, indent=2))
    print("Video/CNN smoke passed; static count 0; not a live exercise validation")
