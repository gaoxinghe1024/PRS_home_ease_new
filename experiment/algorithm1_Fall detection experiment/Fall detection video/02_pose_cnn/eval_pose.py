# -*- coding: utf-8 -*-
"""PCK evaluation for pose models."""

import numpy as np
import torch

from dataset_coco import NUM_KPTS, OUT_H, OUT_W, IMG_H, IMG_W

SH_L, SH_R, HIP_L, HIP_R = 5, 6, 11, 12


@torch.no_grad()
def predict_coords(model, imgs, device):
    model.eval()
    out = model(imgs.to(device))
    hm = out[-1]
    B, C, H, W = hm.shape
    flat = hm.view(B, C, -1)
    conf, idx = flat.max(-1)
    xs = (idx % W).float() * (IMG_W / W)
    ys = (idx // W).float() * (IMG_H / H)
    return torch.stack([xs, ys], -1), conf


def torso_sizes(kpts):
    B = kpts.shape[0]
    sizes = np.zeros(B, dtype=np.float64)
    for b in range(B):
        k = kpts[b]
        vis = k[:, 2] > 0
        if vis[SH_L] and vis[SH_R] and vis[HIP_L] and vis[HIP_R]:
            sh = (k[SH_L, :2] + k[SH_R, :2]) / 2.0
            hp = (k[HIP_L, :2] + k[HIP_R, :2]) / 2.0
            sizes[b] = np.linalg.norm(sh - hp)
        else:
            x, y, v = k[:, 0], k[:, 1], k[:, 2]
            if vis.any():
                xs, ys = x[vis], y[vis]
                sizes[b] = np.hypot(xs.max() - xs.min(), ys.max() - ys.min())
            else:
                sizes[b] = 1.0
    sizes[sizes < 1e-6] = 1.0
    return sizes


@torch.no_grad()
def pck_on_loader(model, loader, device, alphas=(0.2, 0.5), max_batches=None):
    model.eval()
    per_kpt_hit = np.zeros(NUM_KPTS, dtype=np.int64)
    per_kpt_tot = np.zeros(NUM_KPTS, dtype=np.int64)
    hit = {a: 0 for a in alphas}
    tot = 0
    nb = 0
    for batch in loader:
        imgs = batch["img"]
        kpts = batch["kpts"].numpy()
        pred, _ = predict_coords(model, imgs, device)
        pred = pred.cpu().numpy()

        ts = torso_sizes(kpts)[:, None]  # [B,1]
        gt_xy = kpts[:, :, :2]
        gt_v = kpts[:, :, 2]
        d = np.linalg.norm(pred - gt_xy, axis=-1)  # [B,17]

        for a in alphas:
            ok = (d < a * ts) & (gt_v > 0)
            hit[a] += int(ok.sum())
        tot += int((gt_v > 0).sum())

        ok02 = (d < 0.2 * ts) & (gt_v > 0)
        per_kpt_hit += ok02.sum(0).astype(np.int64)
        per_kpt_tot += (gt_v > 0).sum(0).astype(np.int64)

        nb += 1
        if max_batches and nb >= max_batches:
            break

    out = {f"pck@{a}": (hit[a] / tot if tot else 0.0) for a in alphas}
    out["n_points"] = tot
    out["per_kpt"] = (per_kpt_hit / np.maximum(per_kpt_tot, 1)).tolist()
    return out
