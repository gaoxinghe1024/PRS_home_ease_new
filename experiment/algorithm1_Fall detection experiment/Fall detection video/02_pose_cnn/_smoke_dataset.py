"""Smoke test for the COCO dataset pipeline."""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from dataset_coco import (
    load_coco_person_annots,
    CocoPoseDataset,
    IMG_H,
    IMG_W,
    OUT_H,
    OUT_W,
    NUM_KPTS,
)

ann_path = os.path.join(
    ROOT, "data", "coco", "annotations", "person_keypoints_val2017.json"
)
img_root = os.path.join(ROOT, "data", "coco", "val2017")
anns = load_coco_person_annots(ann_path)
print(f"val2017 samples after filtering: {len(anns)}")
ds = CocoPoseDataset(img_root, anns, training=True)
for i in [0, 5, 100]:
    s = ds[i]
    img, hm, vis, kpts = (s["img"], s["heatmap"], s["vis"], s["kpts"])
    assert img.shape == (3, IMG_H, IMG_W), img.shape
    assert hm.shape == (NUM_KPTS, OUT_H, OUT_W), hm.shape
    assert vis.shape == (NUM_KPTS,)
    peak = hm.amax(dim=(1, 2))
    n_vis = int(vis.sum().item())
    n_peak = int((peak > 0.9).sum().item())
    print(
        f"[{i}] img {tuple(img.shape)} peak range [{peak.min():.2f},{peak.max():.2f}] visible points={n_vis} heatmap peaks>0.9={n_peak} pixel range[{img.min():.2f},{img.max():.2f}]"
    )
    assert n_vis == n_peak, f"visible points {n_vis} vs heatmap peaks {n_peak} mismatch"
ds_eval = CocoPoseDataset(img_root, anns, training=False)
n_pos = 0
for i in range(200):
    s = ds_eval[i]
    v = int(s["vis"].sum().item())
    n_peak = int((s["heatmap"].amax(dim=(1, 2)) > 0.9).sum().item())
    assert v > 0, f"eval sample {i} has no visible point (should not happen)"
    assert v == n_peak, f"eval sample {i} vis({v}) != channels with peak>0.9 ({n_peak})"
    n_pos += 1
print(f"eval mode {n_pos}/200 spot-check passed (one heatmap peak per visible point)")
loss_cnt = 0
for i in range(200):
    s = ds[i]
    v = int(s["vis"].sum().item())
    n_peak = int((s["heatmap"].amax(dim=(1, 2)) > 0.8).sum().item())
    assert n_peak <= v, f"train sample {i} peak count ({n_peak}) > vis({v})"
    if v == 0:
        loss_cnt += 1
print(f"train mode, 200 samples: out-of-bounds {loss_cnt}/200 (should be rare)")
assert loss_cnt <= 10, f"train mode: out-of-bounds ratio too high {loss_cnt}/200"
print("SMOKE_OK")
