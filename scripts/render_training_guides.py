"""Render the two leg exercises directly from original training skeletons."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "algorithm2"))
from homeease.features import stable_features
from pipeline.data import project

EDGES = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (0, 6), (1, 7),
         (6, 7), (6, 8), (8, 10), (7, 9), (9, 11)]


def reference_clip(records, skeletons, movement):
    clips = []
    for record in records:
        if record["split"] != "train" or record["incorrect"] or record["movement"] != movement:
            continue
        key = record["id"]
        poses = project(skeletons[f"{key}_skel"])
        phases = skeletons[f"{key}_labels"] % 4
        features = np.array([stable_features(p) for p in poses])
        signal = features[:, 36 if movement == "m02" else 37]
        peak = np.flatnonzero(phases == 2)
        rest = np.flatnonzero(phases == 0)
        if not len(peak) or not np.any(rest < peak[0]) or not np.any(rest > peak[-1]):
            continue
        if movement == "m06":
            # A leg-raise illustration must also show an extended active knee.
            active = np.argmin(features[peak, 34:36], axis=1)
            extended = features[peak, 24 + active] <= -.9
            separated = np.abs(features[peak, 20] - features[peak, 22]) >= .3
            readable = extended & separated
            if np.mean(readable) < .5:
                continue
            peak = peak[readable]
        clips.append((float(np.median(signal[peak])), key, poses, phases, peak, rest))
    median = np.median([c[0] for c in clips])
    return min(clips, key=lambda c: (abs(c[0] - median), c[1]))


def render(poses, indices, bounds):
    canvas = Image.new("RGB", (780, 252), "#f5f8f6")
    draw = ImageDraw.Draw(canvas)
    low, high = bounds
    factor = min(200 / (high[0] - low[0]), 208 / (high[1] - low[1]))
    center = (low + high) / 2
    for panel, index in enumerate(indices):
        q = (poses[index, :, :2] - center) * factor + [panel * 260 + 130, 132]
        draw.line((panel * 260 + 20, 239, panel * 260 + 240, 239), fill="#d0ddd5", width=2)
        for a, b in EDGES:
            color = "#2f8164" if a % 2 == 0 else "#537eaf"
            draw.line([tuple(q[a]), tuple(q[b])], fill=color, width=4)
        for x, y in q:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="white", outline="#4c7766")
        draw.text((panel * 260 + 14, 10), str(panel + 1), fill="#4c7766", font_size=18)
    return canvas


def main():
    records = json.loads((ROOT / "algorithm2/data/processed/records.json").read_text())
    skeletons = np.load(ROOT / "algorithm2/data/processed/skeletons.npz")
    provenance = {}
    for exercise, movement in [("hurdle_step", "m02"), ("standing_leg_raise", "m06")]:
        _, key, poses, phases, peaks, rests = reference_clip(records, skeletons, movement)
        peak = int(peaks[len(peaks) // 2])
        start = max(0, int(rests[rests < peaks[0]][-1]) - 15)
        end = min(len(poses) - 1, int(rests[rests > peaks[-1]][0]) + 15)
        points = poses[start:end + 1, :, :2].reshape(-1, 2)
        bounds = (points.min(0), points.max(0))
        destination = ROOT / f"web/static/media/{exercise}"
        still = render(poses, [start, peak, end], bounds)
        still.save(f"{destination}-guide.png")
        still.crop((260, 0, 520, 252)).save(f"{destination}-pose.png")
        frames = [render(poses, [start, i, end], bounds) for i in range(start, end + 1, 3)]
        frames[0].save(f"{destination}-guide.gif", save_all=True, append_images=frames[1:],
                       duration=[800] + [100] * (len(frames) - 2) + [800], loop=0, optimize=False)
        provenance[exercise] = {"source": key, "split": "train", "incorrect": False,
            "projection_degrees": 45, "source_fps": 30, "animation_fps": 10,
            "start_frame": start, "peak_frame": peak, "end_frame": end,
            "selection": "nearest_median_hold_height_with_extended_visible_active_leg" if movement == "m06" else "nearest_median_training_clip_hold_height",
            "usage": "visual_reference_only_not_inference_input_or_live_accuracy_evidence"}
    (ROOT / "artifacts/exercise_guide_sources.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance))


if __name__ == "__main__":
    main()
