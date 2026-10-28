"""Read-only model/data checks and real HTTP regression checks for Algorithm 2.

Run in a fresh Python process. Runtime files and uploaded test videos use a temporary
directory. No camera or microphone is opened, and no training/report stage is run.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SYSTEM = REPO / "system code"
EXERCISE = HERE
SYSTEM_EXERCISE = SYSTEM / "algorithms/exercise"
sys.path[:0] = [str(SYSTEM), str(EXERCISE)]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def payload(exercise):
    return json.loads((EXERCISE / "examples" / f"{exercise}_replay.json").read_text(encoding="utf-8"))


def request(client, method, path, expected=200, **kwargs):
    response = client.request(method, path, **kwargs)
    require(response.status_code == expected,
            f"{method} {path}: HTTP {response.status_code}, expected {expected}: {response.text[:500]}")
    return response.json() if response.content and "json" in response.headers.get("content-type", "") else response


def run_checks(include_system=True):
    # Force local tool settings to the isolated run, not the user's actual settings.
    with tempfile.TemporaryDirectory(prefix="homeease-algorithm2-") as directory:
        runtime = Path(directory)
        os.environ["YOLO_CONFIG_DIR"] = str(runtime / "ultralytics")
        return _run_checks(runtime, include_system)


def _run_checks(runtime, include_system):
    import numpy as np
    from fastapi.testclient import TestClient
    from homeease.api import create_app
    from homeease.engine import ExerciseSession, Predictor
    from homeease.exercises import EXERCISE_NAMES

    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Algorithm 2 only; real SDK, standalone HTTP and HomeEase HTTP",
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "fastapi",
                      "uvicorn", "httpx", "ultralytics", "opencv-python")},
        "checks": [],
        "not_checked": ["physical camera/microphone capture", "CUDA acceleration",
                        "new home-video/older-adult accuracy", "retraining",
                        "Algorithm 1 and Algorithm 3 availability/accuracy"],
    }

    def check(name, function):
        started = time.perf_counter()
        try:
            result = function()
            entry = {"name": name, "status": "passed", "details": result}
        except Exception as error:
            entry = {"name": name, "status": "failed", "error": str(error),
                     "traceback": traceback.format_exc()}
        entry["seconds"] = round(time.perf_counter() - started, 3)
        report["checks"].append(entry)
        print(f"{entry['status'].upper()}: {name} ({entry['seconds']}s)", flush=True)
        return entry

    def check_data():
        records = json.loads((EXERCISE / "data/processed/records.json").read_text())
        splits = dict(Counter(r["split"] for r in records))
        require(len(records) == 1000 and splits == {"train": 600, "validation": 200, "test": 200},
                f"Unexpected data split: {splits}")
        require(len({r["id"] for r in records}) == len(records), "Duplicate record ids")
        with np.load(EXERCISE / "data/processed/skeletons.npz", allow_pickle=False) as arrays:
            require(len(arrays.files) == len(records) * 2, "Missing/extra skeleton or label arrays")
            for r in records:
                skel, labels = arrays[r["id"] + "_skel"], arrays[r["id"] + "_labels"]
                require(skel.shape == (r["frames"], 22, 3) and labels.shape == (r["frames"],), r["id"])
                require(np.isfinite(skel).all() and np.isin(labels, np.arange(17)).all(), r["id"])
                split = "train" if r["subject"] <= 6 else "validation" if r["subject"] <= 8 else "test"
                require(split == r["split"], "Subject leakage: " + r["id"])
        manifest = json.loads((EXERCISE / "reports/artifact_manifest.json").read_text())
        for entry in manifest:
            path = EXERCISE.joinpath(*entry["file"].replace("\\", "/").split("/"))
            raw = path.read_bytes()
            if entry.get("text_eol") == "lf":
                raw = raw.replace(b"\r\n", b"\n")
            require(len(raw) == entry["bytes"] and hashlib.sha256(raw).hexdigest() == entry["sha256"],
                    "Artifact mismatch: " + entry["file"])
        # Confirm the exporter survives a Windows text checkout while a binary
        # checkpoint still gets an exact-byte hash.
        from pipeline.common import artifact_manifest
        fixture = runtime / "manifest-fixture"
        (fixture / "artifacts").mkdir(parents=True)
        config = fixture / "artifacts/selected.json"
        config.write_bytes(b'{\n  "model": "fixture"\n}')
        weight = fixture / "artifacts/fixture.pt"
        weight.write_bytes(b"binary\r\ncheckpoint")
        before = artifact_manifest(fixture)
        config.write_bytes(config.read_bytes().replace(b"\n", b"\r\n"))
        require(artifact_manifest(fixture) == before, "Manifest changes under CRLF")
        binary_entry = next(x for x in before if x["file"].endswith(".pt"))
        require(binary_entry["sha256"] == hashlib.sha256(weight.read_bytes()).hexdigest(), "Binary hash normalized")
        # The experiment is a standalone snapshot. Confirm that this inspection
        # starts with the same implementation/data/weights as the deployed copy.
        snapshot_files = 0
        if include_system:
            for folder in ("homeease", "pipeline", "artifacts", "data/processed", "examples"):
                for source in sorted((EXERCISE / folder).rglob("*")):
                    if not source.is_file() or "__pycache__" in source.parts:
                        continue
                    relative = source.relative_to(EXERCISE)
                    target = SYSTEM_EXERCISE / relative
                    require(target.is_file() and hashlib.sha256(source.read_bytes()).digest() ==
                            hashlib.sha256(target.read_bytes()).digest(), "System/snapshot differ: " + str(relative))
                    snapshot_files += 1
        return {"records": len(records), "splits": splits, "verified_artifacts": len(manifest),
                "raw_data_present": (EXERCISE / "data/raw/uiprmd/manifest.json").is_file(),
                "manifest_lf_crlf_regression": True, "system_snapshot_files_compared": snapshot_files}

    check("data, subject split and delivered artifact hashes", check_data)
    predictor = Predictor(device="cpu")
    report["model"] = predictor.selected
    sdk = {}

    def check_sdk():
        existing = {r["exercise"]: r["count"] for r in json.loads((EXERCISE / "reports/replay_sessions.json").read_text())}
        for exercise in EXERCISE_NAMES:
            data = payload(exercise)
            session = ExerciseSession(predictor, exercise=exercise, target_reps=data["reference_count"])
            for frame in data["frames"]:
                session.update(frame["timestamp_ms"], frame["keypoints"], frame.get("person_id", "primary"))
            result = session.control("end")
            require(result["count"] == existing[exercise], f"{exercise}: replay differs from delivered result")
            sdk[exercise] = {"count": result["count"], "reference_count": data["reference_count"],
                             "frames": len(data["frames"])}
        return sdk

    check("four full replays through the original CPU SDK", check_sdk)

    def check_http():
        counts = {}
        with TestClient(create_app(predictor)) as client:
            request(client, "GET", "/health")
            actual_schemas = client.get("/openapi.json").json()["components"]["schemas"]
            for directory in (EXERCISE, SYSTEM_EXERCISE) if include_system else (EXERCISE,):
                saved_schemas = json.loads((directory / "docs/openapi.json").read_text())["components"]["schemas"]
                for schema in ("CreateSession", "SessionState", "BatchResult"):
                    require(saved_schemas[schema]["properties"] == actual_schemas[schema]["properties"],
                            "Stale OpenAPI properties: " + schema)
            for exercise in EXERCISE_NAMES:
                data = payload(exercise)
                state = request(client, "POST", "/v1/sessions", expected=201,
                                json={"exercise": exercise, "target_reps": data["reference_count"]})
                sid = state["session_id"]
                for start in range(0, len(data["frames"]), 100):
                    body = {"request_id": f"batch-{start}", "frames": data["frames"][start:start + 100]}
                    result = request(client, "POST", f"/v1/sessions/{sid}/frames", json=body)
                    if start == 0:
                        retry = request(client, "POST", f"/v1/sessions/{sid}/frames", json=body)
                        require(retry == result, "Retry changed the result")
                state = request(client, "POST", f"/v1/sessions/{sid}/control", json={"action": "end"})
                require(state["count"] == sdk[exercise]["count"], exercise + ": SDK/HTTP mismatch")
                counts[exercise] = state["count"]
                request(client, "DELETE", f"/v1/sessions/{sid}", expected=204)
                request(client, "GET", f"/v1/sessions/{sid}", expected=404)
            sid = request(client, "POST", "/v1/sessions", expected=201, json={"exercise": "squat"})["session_id"]
            first = {"request_id": "contract", "frames": payload("squat")["frames"][:2]}
            request(client, "POST", f"/v1/sessions/{sid}/frames", json=first)
            conflict = {**first, "frames": first["frames"][:1]}
            request(client, "POST", f"/v1/sessions/{sid}/frames", expected=409, json=conflict)
            request(client, "POST", f"/v1/sessions/{sid}/frames", expected=409,
                    json={**first, "request_id": "old-timestamps"})
            request(client, "POST", f"/v1/sessions/{sid}/frames", expected=422,
                    json={"request_id": "unordered", "frames": first["frames"][::-1]})
            request(client, "POST", "/v1/sessions", expected=422, json={"exercise": "unknown"})
            request(client, "POST", "/v1/sessions", expected=422,
                    json={"exercise": "squat", "action_scope": "training", "counting_policy": "pose_cycle_v2"})
            request(client, "POST", f"/v1/sessions/{sid}/control", json={"action": "pause"})
            request(client, "POST", f"/v1/sessions/{sid}/frames", expected=409,
                    json={"request_id": "paused", "frames": payload("squat")["frames"][2:4]})
            request(client, "POST", f"/v1/sessions/{sid}/control", json={"action": "resume"})
            request(client, "POST", f"/v1/sessions/{sid}/control", json={"action": "end"})
        return {"counts": counts, "retry_idempotency": True, "conflict_and_validation": True,
                "pause_resume_end": True, "delete": True, "saved_openapi_properties": True}

    check("standalone HTTP replay and session contract", check_http)

    def check_interruptions():
        frames = payload("squat")["frames"]
        s = ExerciseSession(predictor, exercise="squat")
        s.update(0, frames[0]["keypoints"])
        state = s.update(500, frames[1]["keypoints"])
        require(any(x["reason"] == "timestamp_gap" for x in state["interruptions"]), "Gap not recorded")
        state = s.update(533, frames[2]["keypoints"], "another-person")
        require(any(x["reason"] == "person_changed" for x in state["interruptions"]), "Person switch not recorded")
        missing = np.asarray(frames[0]["keypoints"]).copy()
        missing[15, 2] = 0
        state = s.update(566, missing, "another-person")
        require(state["quality"] == "insufficient_pose" and state["count"] == 0, "Missing joint not rejected")
        still = ExerciseSession(predictor, exercise="squat")
        for index in range(120):
            state = still.update(index * 1000 / 30, frames[0]["keypoints"])
        require(state["count"] == 0, "Still input counted a repetition")
        return {"timestamp_gap": True, "person_change": True, "missing_joint": True, "still_count": state["count"]}

    check("interruption and still-input safeguards", check_interruptions)

    if include_system:
        check("real HomeEase server, worker, replay, controls and upload", lambda: check_system(runtime, sdk))
    else:
        report["not_checked"].append("HomeEase system HTTP (--skip-system)")
    report["passed"] = all(x["status"] == "passed" for x in report["checks"])
    return report


def check_system(runtime, sdk):
    import cv2
    import httpx
    import numpy as np

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = os.environ.copy()
    env.pop("HOMEEASE_PASSWORD", None)
    env.update(HOMEEASE_SYSTEM_DATA=str(runtime / "system"), HOMEEASE_EXERCISE_DEVICE="cpu",
               HOMEEASE_VIDEO_DNN="opencv", PYTHONIOENCODING="utf-8")
    log_path = runtime / "server.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, str(SYSTEM / "run.py"), "--port", str(port)],
                                   cwd=SYSTEM, env=env, stdout=log, stderr=log,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=180) as client:
                deadline = time.monotonic() + 150
                while True:
                    require(process.poll() is None, "Server exited: " + log_path.read_text(encoding="utf-8")[-4000:])
                    try:
                        health = request(client, "GET", "/api/health")
                        if any(m["id"] == "exercise" and m["state"] == "ready" for m in health["modules"]):
                            break
                    except httpx.TransportError:
                        pass
                    require(time.monotonic() < deadline, "Exercise startup timeout: " + log_path.read_text(encoding="utf-8")[-4000:])
                    time.sleep(.3)
                pages = ["/", "/home", "/elder", "/api/exercises", "/static/app.js", "/static/exercise.css"]
                for path in pages:
                    request(client, "GET", path)
                counts = {}
                for exercise, expected in sdk.items():
                    state = request(client, "POST", "/api/sessions", expected=201,
                                    json={"exercise": exercise, "source": "replay", "target_reps": expected["reference_count"]})
                    sid = state["session_id"]
                    state = wait_session(client, sid)
                    require(state["count"] == expected["count"], exercise + ": HomeEase/SDK count mismatch")
                    trace = request(client, "GET", f"/api/sessions/{sid}/diagnostics.json")
                    require(trace["frames"] and trace["state"]["count"] == expected["count"], "Missing trace")
                    exported = request(client, "GET", f"/api/sessions/{sid}/export.json")
                    require(exported["status"] == "ended" and exported["count"] == expected["count"], "Bad export")
                    counts[exercise] = state["count"]
                policies = {}
                for exercise in sdk:
                    state = request(client, "POST", "/api/sessions", expected=201,
                                    json={"exercise": exercise, "source": "camera", "capture": "client"})
                    sid = state["session_id"]
                    expected = "pose_cycle_v2" if exercise == "chair_sit_to_stand" else "pose_cycle_v1" if exercise == "standing_leg_raise" else "original"
                    require(state["action_scope"] == "training" and state["counting_policy"] == expected, "Wrong deployment policy")
                    policies[exercise] = {k: state[k] for k in ("counting_policy", "inference_views", "phase_policy")}
                    request(client, "POST", f"/api/sessions/{sid}/control", json={"action": "pause"})
                    request(client, "POST", f"/api/sessions/{sid}/control", json={"action": "resume"})
                    request(client, "POST", f"/api/sessions/{sid}/control", json={"action": "end"})
                # A generated empty scene exercises real video decoding, pose inference
                # and the upload worker. It is explicitly a negative functional test.
                video_path = runtime / "empty-scene.avi"
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
                require(writer.isOpened(), "Cannot encode the negative test video")
                try:
                    for _ in range(12):
                        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
                finally:
                    writer.release()
                upload = request(client, "POST", "/api/uploads/video",
                                 files={"file": ("empty-scene.avi", video_path.read_bytes(), "video/x-msvideo")})
                state = request(client, "POST", "/api/sessions", expected=201,
                                json={"exercise": "squat", "source": "upload", "upload_id": upload["upload_id"]})
                state = wait_session(client, state["session_id"])
                require(state["count"] == 0 and state["counting_diagnostics"]["observed_frames"] == 12,
                        "Empty scene was counted or video frames were dropped")
                request(client, "GET", "/api/reports?mode=replay")
                result = {"pages": pages, "counts": counts, "deployment_policies": policies,
                          "upload_negative": {"frames": 12, "count": state["count"]},
                          "exercise_worker": next(m for m in health["modules"] if m["id"] == "exercise")}
                # Absolute PIDs/timestamps are transient; don't publish them as artifacts.
                result["exercise_worker"].pop("pid", None)
                result["exercise_worker"].pop("last_call", None)
                return result
        except Exception as error:
            raise RuntimeError(f"{error}\nServer log (tail):\n{log_path.read_text(encoding='utf-8')[-5000:]}") from error
        finally:
            (runtime / "system" / "stop.request").write_text("stop", encoding="ascii")
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                import psutil
                try:
                    children = psutil.Process(process.pid).children(recursive=True)
                except psutil.NoSuchProcess:
                    children = []
                for child in children:
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                process.terminate()
                process.wait(timeout=10)


def wait_session(client, sid):
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        state = request(client, "GET", f"/api/sessions/{sid}")
        if state["status"] in {"ended", "interrupted"}:
            require(state["status"] == "ended", "Session interrupted: " + json.dumps(state)[:1200])
            return state
        time.sleep(.25)
    raise AssertionError("Session timeout: " + sid)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-system", action="store_true", help="only check data, SDK and standalone HTTP")
    parser.add_argument("--output", type=Path, default=HERE / "results/runtime_check.json")
    args = parser.parse_args()
    report = run_checks(not args.skip_system)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {args.output}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
