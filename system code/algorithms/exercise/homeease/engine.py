"""Causal phase inference and the counting session shared by SDK, replay, video and HTTP."""
import json
import threading
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .counter import CycleCounter, PHASES
from .features import FeatureExtractor
from .pose_cycle import PoseCyclePhases
from .models import build_model
from .exercises import EXERCISES, EXERCISE_NAMES, decode

ROOT = Path(__file__).resolve().parents[1]


class Predictor:
    def __init__(self, artifact_dir=None, device="cpu"):
        self.directory = Path(artifact_dir or ROOT / "artifacts")
        self.selected = json.loads((self.directory / "selected.json").read_text())
        self.device = torch.device(device)
        self.members = []
        self.lock = threading.RLock()
        torch.set_num_threads(4)
        for filename in self.selected["members"]:
            package = torch.load(self.directory / filename, map_location="cpu", weights_only=True)
            model = build_model(package["config"], package["input_dim"])
            model.load_state_dict(package["state_dict"])
            model.to(self.device).eval()
            self.members.append({"model": model, "mode": package["config"]["features"],
                                 "context": package.get("context_frames", 64),
                                 "mean": np.asarray(package["mean"], dtype=np.float32),
                                 "std": np.asarray(package["std"], dtype=np.float32)})

    def stream(self):
        return PhaseStream(self)


class PhaseStream:
    def __init__(self, predictor):
        self.predictor = predictor
        self.reset()

    def reset(self):
        self.extractors = [FeatureExtractor(m["mode"]) for m in self.predictor.members]
        self.buffers = [deque(maxlen=m["context"]) for m in self.predictor.members]

    def _window(self, index, pose, timestamp):
        member = self.predictor.members[index]
        feature = self.extractors[index].update(pose, timestamp)
        buffer = self.buffers[index]
        buffer.append((feature - member["mean"]) / member["std"])
        x = np.asarray(list(buffer), dtype=np.float32)
        if len(x) < member["context"]:
            x = np.pad(x, ((member["context"] - len(x), 0), (0, 0)), mode="edge")
        return x

    def update(self, pose, timestamp):
        probs = []
        with self.predictor.lock, torch.inference_mode():
            for index, member in enumerate(self.predictor.members):
                x = self._window(index, pose, timestamp)
                logits = member["model"](torch.from_numpy(x[None]).to(self.predictor.device))[:, -1]
                probs.append(logits.softmax(-1).cpu().numpy()[0])
        return np.mean(probs, axis=0)

    def update_many(self, observations):
        """Run available causal windows together, without waiting for a frame."""
        if not observations:
            return []
        probabilities = []
        with self.predictor.lock, torch.inference_mode():
            for index, member in enumerate(self.predictor.members):
                windows = np.asarray([self._window(index, pose, timestamp)
                                      for pose, timestamp in observations], dtype=np.float32)
                logits = member["model"](torch.from_numpy(windows).to(self.predictor.device))[:, -1]
                probabilities.append(logits.softmax(-1).cpu().numpy())
        return np.mean(probabilities, axis=0)


class MirroredPhaseStream:
    """Average the original view and its mirror, the pair the training augmentation used."""
    def __init__(self, predictor):
        self.original = predictor.stream()
        self.mirrored = predictor.stream()

    def reset(self):
        self.original.reset()
        self.mirrored.reset()

    def update(self, pose, timestamp):
        return self.update_many([(pose, timestamp)])[0]

    def update_many(self, observations):
        if not observations:
            return []
        views = []
        for pose, timestamp in observations:
            reflected = pose[[1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]].copy()
            reflected[:, 0] *= -1
            views.append((pose, reflected, timestamp))
        streams = (self.original, self.mirrored)
        if any(type(stream) is not PhaseStream for stream in streams):
            return [(self.original.update(pose, timestamp) + self.mirrored.update(reflected, timestamp)) / 2
                    for pose, reflected, timestamp in views]
        predictor = self.original.predictor
        probabilities = []
        # Batch available ticks without using future frames.
        with predictor.lock, torch.inference_mode():
            for index, member in enumerate(predictor.members):
                batch = [stream._window(index, view, timestamp)
                         for pose, reflected, timestamp in views
                         for stream, view in zip(streams, (pose, reflected))]
                logits = member["model"](torch.from_numpy(np.asarray(batch)).to(predictor.device))[:, -1]
                probabilities.append(logits.softmax(-1).cpu().numpy())
        paired = np.mean(probabilities, axis=0).reshape(len(views), 2, -1)
        return (paired[:, 0] + paired[:, 1]) / 2


class ExerciseSession:
    def __init__(self, predictor, session_id="local", target_reps=10, exercise="chair_sit_to_stand", action_scope="open", phase_policy=None, counting_policy="original"):
        if exercise not in EXERCISES:
            raise ValueError("UNSUPPORTED_EXERCISE")
        if not 1 <= target_reps <= 10000:
            raise ValueError("INVALID_TARGET")
        if action_scope not in {"open", "training"}:
            raise ValueError("INVALID_ACTION_SCOPE")
        self.exercise = exercise
        self.action_scope = action_scope
        if counting_policy not in {"original", "pose_cycle_v1", "pose_cycle_v2"} or (
            counting_policy != "original" and (action_scope != "training" or
                exercise not in {"chair_sit_to_stand", "standing_leg_raise"})) or (
            counting_policy == "pose_cycle_v2" and exercise != "chair_sit_to_stand"):
            raise ValueError("INVALID_COUNTING_POLICY")
        self.counting_policy = counting_policy
        self.pose_phases = PoseCyclePhases(exercise) if counting_policy != "original" else None
        self.counting_phase = "other"
        # Allow slower seated transitions while retaining interruption checks.
        self.max_input_gap_s = .4 if counting_policy == "pose_cycle_v2" else .25
        self.predictor, self.session_id, self.target_reps = predictor, session_id, target_reps
        use_mirror = action_scope == "training" and exercise in {"standing_leg_raise", "hurdle_step"}
        use_action_hold = action_scope == "training" and exercise == "hurdle_step"
        self.phase_policy = phase_policy or ("action_gap_hold_v1" if use_action_hold else "original")
        if self.phase_policy not in {"original", "action_gap_hold_v1"} or (self.phase_policy != "original" and not use_action_hold):
            raise ValueError("INVALID_PHASE_POLICY")
        self.last_supported_action_tick = None
        self.inference_views = "mirror_average" if use_mirror else "original"
        self.stream = MirroredPhaseStream(predictor) if self.inference_views == "mirror_average" else predictor.stream()
        self.counter = CycleCounter(min_dwell_s=.1 if self.pose_phases else predictor.selected["min_dwell_s"])
        self.status = "running"
        self.quality = "waiting_for_input"
        self.last_timestamp = self.first_timestamp = None
        self.last_pose = self.next_tick = None
        self.person_id = None
        self.phase, self.confidence = "other", 0.0
        self.probabilities = np.zeros(5)
        self.action_probabilities = np.zeros(5)
        self.recognized_exercise = "other"
        self.active_duration_s = 0.0
        self.valid_duration_s = 0.0
        self.interruptions = []
        self.skip_next_interval = False
        self.lock = threading.RLock()

    def interrupt(self, reason, timestamp=None):
        if self.quality != reason:
            self.interruptions.append({"reason": reason, "timestamp_ms": timestamp})
        self.quality = reason
        self.counter.interrupt()
        if self.pose_phases:
            self.pose_phases.reset()
        self.counting_phase = "other"
        self.last_supported_action_tick = None
        self.stream.reset()
        self.last_pose = self.next_tick = None
        self.phase, self.confidence = "other", 0.0
        self.probabilities = np.zeros(5)
        self.action_probabilities = np.zeros(5)
        self.recognized_exercise = "other"

    def update(self, timestamp_ms, keypoints, person_id="primary"):
        with self.lock:
            if self.status != "running":
                raise ValueError("SESSION_NOT_RUNNING")
            if not np.isfinite(timestamp_ms) or not 0 <= timestamp_ms <= 1e15:
                raise ValueError("INVALID_TIMESTAMP")
            if self.last_timestamp is not None and timestamp_ms <= self.last_timestamp:
                raise ValueError("TIMESTAMP_NOT_INCREASING")
            pose = np.asarray(keypoints, dtype=np.float32)
            if pose.shape == (17, 3):
                pose = pose[5:17]
            if pose.shape != (12, 3) or not np.isfinite(pose).all():
                raise ValueError("INVALID_KEYPOINTS")
            if np.max(np.abs(pose[:, :2])) > 1e7:
                raise ValueError("INVALID_KEYPOINTS")
            if np.any((pose[:, 2] < 0) | (pose[:, 2] > 1)):
                raise ValueError("INVALID_CONFIDENCE")
            first = self.first_timestamp is None
            if first:
                self.first_timestamp = timestamp_ms
            previous_timestamp = self.last_timestamp
            self.last_timestamp = timestamp_ms
            t = (timestamp_ms - self.first_timestamp) / 1000
            gap_s = 0 if previous_timestamp is None else (timestamp_ms - previous_timestamp) / 1000
            if gap_s > self.max_input_gap_s:
                self.interrupt("timestamp_gap", timestamp_ms)
            elif previous_timestamp is not None and not self.skip_next_interval:
                self.active_duration_s += gap_s
            self.skip_next_interval = False
            if self.person_id is not None and person_id != self.person_id:
                self.interrupt("person_changed", timestamp_ms)
            self.person_id = person_id
            required = [0, 1, 6, 7, 8, 9, 10, 11]
            if np.min(pose[required, 2]) < .3:
                self.interrupt("insufficient_pose", timestamp_ms)
                return self.snapshot()
            leg_extent = sum(np.linalg.norm(pose[a, :2] - pose[b, :2]) for a, b in [(6, 8), (8, 10), (7, 9), (9, 11)])
            torso_extent = np.linalg.norm((pose[0, :2] + pose[1, :2]) / 2 - (pose[6, :2] + pose[7, :2]) / 2)
            if leg_extent < 1e-4 or torso_extent < 1e-4:
                self.interrupt("degenerate_pose", timestamp_ms)
                return self.snapshot()
            if self.last_pose is None:
                self.next_tick = t
                self.last_pose = (t, pose.copy())
            previous_t, previous_pose = self.last_pose
            observations = []
            while self.next_tick <= t + 1e-8:
                ratio = 1 if t <= previous_t else (self.next_tick - previous_t) / (t - previous_t)
                interpolated = previous_pose * (1 - ratio) + pose * ratio
                observations.append((interpolated, self.next_tick))
                self.next_tick += 1 / 30
            update_many = getattr(self.stream, "update_many", None)
            if type(self.stream) is PhaseStream and self.counting_policy != "pose_cycle_v2":
                update_many = None
            posteriors = (update_many(observations) if update_many is not None else
                          (self.stream.update(pose, tick) for pose, tick in observations))
            for (observed_pose, tick), p in zip(observations, posteriors):
                phase, canonical, actions = decode(p, self.exercise, self.action_scope)
                hold_action_gap = False
                if self.phase_policy == "action_gap_hold_v1":
                    target = EXERCISE_NAMES.index(self.exercise)
                    support = actions[target] / max(actions[:4].sum(), 1e-8)
                    if support >= .15:
                        self.last_supported_action_tick = tick
                    hold_action_gap = (canonical[4] >= .5 and self.counter.stage > 0
                        and self.last_supported_action_tick is not None
                        and tick - self.last_supported_action_tick <= .15)
                self.action_probabilities = actions
                action = int(actions.argmax())
                self.recognized_exercise = EXERCISE_NAMES[action] if action < 4 else "other"
                self.probabilities = canonical
                self.phase = EXERCISES[self.exercise]["phases"][phase] if phase < 4 else "other"
                self.confidence = float(p.max())
                if self.pose_phases:
                    self.counting_phase = self.pose_phases.update(observed_pose, tick, self.counter.stage)
                    self.counter.update(self.counting_phase, tick)
                elif hold_action_gap:
                    # Hold the current stage without advancing the cycle.
                    self.counter.update(PHASES[self.counter.stage], tick)
                elif self.predictor.selected.get("decoder_strategy") in ["ordered_posterior", "ordered_transition"]:
                    self.counter.update_probabilities(canonical, tick, self.predictor.selected["phase_evidence_threshold"],
                                                      infer_return=self.predictor.selected["decoder_strategy"] == "ordered_transition")
                else:
                    self.counter.update(PHASES[phase] if self.confidence >= self.predictor.selected["confidence_threshold"] else "other", tick)
            if previous_timestamp is not None and gap_s <= self.max_input_gap_s and self.quality == "good":
                self.valid_duration_s += gap_s
            self.quality = "good"
            self.last_pose = (t, pose.copy())
            return self.snapshot()

    def control(self, action):
        with self.lock:
            if self.status == "ended":
                if action == "end":
                    return self.snapshot()
                raise ValueError("SESSION_ENDED")
            if action == "pause":
                self.status = "paused"
                self.interrupt("paused", self.last_timestamp)
                self.skip_next_interval = True
            elif action == "resume":
                self.status = "running"
                self.interrupt("waiting_for_input", self.last_timestamp)
                self.skip_next_interval = True
            elif action == "end":
                self.status = "ended"
                self.counter.interrupt()
            else:
                raise ValueError("INVALID_ACTION")
            return self.snapshot()

    def snapshot(self):
        phase = self.phase
        if self.pose_phases:
            index = PHASES.index(self.counting_phase)
            phase = EXERCISES[self.exercise]["phases"][index] if index < 4 else "other"
            if phase == "other" and self.pose_phases.posture in {"standing", "sitting"}:
                phase = self.pose_phases.posture
        return {"api_version": "v1", "session_id": self.session_id, "status": self.status,
                "counting_policy": self.counting_policy, "model_phase": self.phase,
                "setup_pose": self.pose_phases.posture if self.pose_phases else None,
                "max_input_gap_ms": self.max_input_gap_s * 1000,
                "exercise": self.exercise, "action_scope": self.action_scope, "inference_views": self.inference_views,
                "phase_policy": self.phase_policy,
                "recognized_exercise": self.recognized_exercise,
                "exercise_probabilities": dict(zip(EXERCISE_NAMES + ["other"], [float(v) for v in self.action_probabilities])),
                "phase": phase, "phase_confidence": self.confidence,
                "phase_probabilities": dict(zip(EXERCISES[self.exercise]["phases"] + ["other"], [float(v) for v in self.probabilities])),
                "count": self.counter.count, "counter_stage": self.counter.stage, "target_reps": self.target_reps,
                "progress": min(self.counter.count / self.target_reps, 1), "target_completed": self.counter.count >= self.target_reps,
                "elapsed_s": 0 if self.first_timestamp is None else (self.last_timestamp - self.first_timestamp) / 1000,
                "active_duration_s": self.active_duration_s, "valid_duration_s": self.valid_duration_s,
                "quality": self.quality, "last_timestamp_ms": self.last_timestamp,
                "interruptions": list(self.interruptions), "cycles": list(self.counter.events),
                "model_id": self.predictor.selected["model"], "model_validation": "public_mocap_weak_phase_reconstructed_replay"}
