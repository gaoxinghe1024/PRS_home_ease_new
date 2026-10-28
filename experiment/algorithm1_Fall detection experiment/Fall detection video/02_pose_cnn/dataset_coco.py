# -*- coding: utf-8 -*-
"""COCO keypoints dataset with affine cropping and heatmap targets."""

import os
import json
import math
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
IMG_H, IMG_W = 256, 192
OUT_H, OUT_W = 64, 48
HM_SIGMA = 2.0
BBOX_EXPAND = 1.25
NUM_KPTS = 17

COCO_KPT_NAMES = [
    "nose",
    "l_eye",
    "r_eye",
    "l_ear",
    "r_ear",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_hip",
    "r_hip",
    "l_knee",
    "r_knee",
    "l_ankle",
    "r_ankle",
]
FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
FLIP_IDX = np.arange(NUM_KPTS)
for a, b in FLIP_PAIRS:
    FLIP_IDX[a] = b
    FLIP_IDX[b] = a

COCO_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def load_coco_person_annots(json_path, min_kpts=6):
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    imgs = {im["id"]: im for im in coco["images"]}
    samples = []
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        if ann.get("num_keypoints", 0) < min_kpts:
            continue
        kpts = np.asarray(ann["keypoints"], dtype=np.float32).reshape(NUM_KPTS, 3)
        x, y, w, h = ann["bbox"]
        if w <= 1 or h <= 1:
            continue
        samples.append(
            {
                "image_id": ann["image_id"],
                "file_name": imgs[ann["image_id"]]["file_name"],
                "bbox": np.asarray([x, y, w, h], dtype=np.float32),
                "keypoints": kpts,
            }
        )
    return samples


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def _affine_matrix(center, scale, rot_deg, out_w, out_h):
    cos_v = math.cos(math.radians(rot_deg))
    sin_v = math.sin(math.radians(rot_deg))
    sx = scale * cos_v
    sy = scale * cos_v
    shear_x = -scale * sin_v
    shear_y = scale * sin_v
    tx = out_w / 2.0 - sx * center[0] - shear_x * center[1]
    ty = out_h / 2.0 - sy * center[1] - shear_y * center[0]
    return np.asarray([[sx, shear_x, tx], [shear_y, sy, ty]], dtype=np.float64)


def _affine_kpts(kpts, M):
    xy = kpts[:, :2]
    ones = np.ones((len(xy), 1), dtype=np.float64)
    out = np.concatenate([xy, ones], axis=1) @ M.T
    return np.concatenate([out, kpts[:, 2:3]], axis=1).astype(np.float32)


def affine_crop(img, bbox, out_h, out_w, expand=BBOX_EXPAND, scale=1.0, rot=0.0):
    x, y, w, h = [float(v) for v in bbox]
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * expand
    src_scale = out_w / side
    total = src_scale * scale
    M = _affine_matrix((cx, cy), total, rot, out_w, out_h)
    warped = cv2.warpAffine(
        img,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(114, 114, 114),
    )
    return warped, M


def _color_jitter(img, rng):
    img = img.astype(np.float32)
    alpha = rng.uniform(0.6, 1.4)
    beta = rng.uniform(-18, 18)
    img = img * alpha + beta
    hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(
        hsv[..., 1].astype(np.float32) * rng.uniform(0.6, 1.4), 0, 255
    ).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _gaussian_heatmap(h, w, cx, cy, sigma):
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return np.exp(-d2 / (2.0 * sigma * sigma))


def make_heatmaps(kpts_affine, out_h=OUT_H, out_w=OUT_W, sigma=HM_SIGMA):
    hm = np.zeros((NUM_KPTS, out_h, out_w), dtype=np.float32)
    vis = np.zeros(NUM_KPTS, dtype=np.float32)
    for i in range(NUM_KPTS):
        x, y, v = kpts_affine[i]
        if v > 0 and -out_w <= x < 2 * out_w and -out_h <= y < 2 * out_h:
            if 0 <= x < out_w and 0 <= y < out_h:
                hm[i] = _gaussian_heatmap(out_h, out_w, x, y, sigma)
                vis[i] = 1.0
    return hm, vis


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CocoPoseDataset(Dataset):

    def __init__(self, img_root, anns, training=True):
        self.img_root = img_root
        self.anns = anns
        self.training = training

    def __len__(self):
        return len(self.anns)

    def _load_img(self, file_name):
        path = os.path.join(self.img_root, file_name)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            path2 = os.path.join(self.img_root, file_name.split("/")[-1])
            img = cv2.imread(path2, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"COCO image not found: {file_name}")
        return img

    def __getitem__(self, idx):
        a = self.anns[idx]
        img = self._load_img(a["file_name"])
        kpts = a["keypoints"].copy()
        bbox = a["bbox"].copy()

        if self.training:
            rng = random.Random()
            rot = rng.uniform(-30, 30)
            scale = rng.uniform(0.75, 1.25)
            flip = rng.random() < 0.5
        else:
            rot, scale, flip = 0.0, 1.0, False

        if flip:
            img = img[:, ::-1, :].copy()
            W = img.shape[1]
            kpts[:, 0] = W - 1 - kpts[:, 0]
            kpts = kpts[FLIP_IDX]
            bbox[0] = W - 1 - (bbox[0] + bbox[2])

        img_warp, M = affine_crop(img, bbox, IMG_H, IMG_W, scale=scale, rot=rot)
        kpts_warp = _affine_kpts(kpts, M)

        if self.training:
            img_warp = _color_jitter(img_warp, random)

        hm, vis = make_heatmaps(kpts_warp * (OUT_W / IMG_W))

        img_t = torch.from_numpy(np.ascontiguousarray(img_warp[:, :, ::-1]))  # BGR->RGB
        img_t = img_t.permute(2, 0, 1).float() / 255.0
        return {
            "img": img_t,
            "heatmap": torch.from_numpy(hm),
            "vis": torch.from_numpy(vis),
            "kpts": torch.from_numpy(kpts_warp),
            "idx": idx,
        }


def collate_keep(batch):
    return torch.utils.data.default_collate(batch)
