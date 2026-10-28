"""Visible endpoint evidence for the two exercises counted from explicit camera training.

The learned action/phase posteriors stay untouched; this policy only supplies
movement phases to the ordered counter, for exercises whose transitions the
camera-domain classifier represents poorly.
"""
import numpy as np

from .features import cosine


class PoseCyclePhases:
    def __init__(self, exercise):
        self.exercise = exercise
        self.reset()

    def reset(self):
        self.posture = "waiting"
        self.baseline = None
        self.side = None
        self.peak = None
        self.rest_since = None

    def update(self, pose, timestamp, stage):
        p = np.asarray(pose)[:, :2]
        torso = max(float(np.linalg.norm((p[0] + p[1] - p[6] - p[7]) / 2)), 1e-4)
        knees = np.degrees(np.arccos(np.clip(
            [cosine(p[6], p[8], p[10]), cosine(p[7], p[9], p[11])], -1, 1)))
        height = float((p[8, 1] + p[9, 1] - p[6, 1] - p[7, 1]) / (2 * torso))
        ankle_delta = float(abs(p[10, 1] - p[11, 1]) / torso)
        if self.exercise == "chair_sit_to_stand":
            rest = max(knees) <= 135 and height <= .55 and ankle_delta <= .3
            level = float(np.mean(knees))
            peak = min(knees) >= 155 and height >= .6
            self.posture = "sitting" if rest else "standing" if peak else "transitioning"
            departure = self.baseline is not None and level >= self.baseline + 12
            returning = self.peak is not None and level <= self.peak - 12
        else:
            rest = min(knees) >= 150 and ankle_delta <= .12 and height >= .55
            if stage <= 0:
                self.side = int(np.argmin(p[[10, 11], 1])) if ankle_delta >= .15 else None
            side = self.side
            # One leg stays selected until both feet are back on the floor.
            level = float((p[11 - side, 1] - p[10 + side, 1]) / torso) if side is not None else 0.
            straight = side is not None and knees[side] >= 135 and knees[1 - side] >= 140
            peak = straight and level >= .3
            departure = straight and level >= .15
            returning = self.peak is not None and level <= self.peak - .12
            if side is not None and (level < -.15 or (level >= .2 and knees[side] < 120)):
                # A bent knee cannot finish a straight-leg repetition.
                return "other"

        if stage < 0:
            self.peak = None
            if not rest:
                self.rest_since = None
                return "other"
            if self.rest_since is None:
                self.rest_since = timestamp
            if timestamp - self.rest_since < .15:
                return "other"
            self.baseline = level
            return "rest"
        self.rest_since = None
        if stage == 0:
            if rest:
                self.baseline = level
                self.peak = None
                return "rest"
            return "outbound" if departure else "rest"
        if stage == 1:
            if rest:
                self.reset()
                return "other"
            if peak:
                self.peak = level
                return "hold"
            return "outbound"
        if stage == 2:
            self.peak = max(self.peak if self.peak is not None else level, level)
            return "return" if returning or rest else "hold"
        if rest:
            return "rest"
        return "return"
