"""Diagnostic script for heatmap peak regression."""

import os, sys, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import build_model, _init_from_scratch
from dataset_coco import (
    load_coco_person_annots,
    CocoPoseDataset,
    _gaussian_heatmap,
    OUT_H,
    OUT_W,
    IMG_H,
    IMG_W,
)

torch.manual_seed(0)
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
anns = load_coco_person_annots(
    os.path.join(ROOT, "data", "coco", "annotations", "person_keypoints_val2017.json")
)
ds = CocoPoseDataset(
    os.path.join(ROOT, "data", "coco", "val2017"), anns[:16], training=False
)
imgs = torch.stack([ds[i]["img"] for i in range(16)]).cuda()
fixed = torch.zeros(17, OUT_H, OUT_W)
for c in range(17):
    fixed[c] = torch.from_numpy(_gaussian_heatmap(OUT_H, OUT_W, 24.0, 32.0, 2.0))
tgt = fixed.unsqueeze(0).repeat(16, 1, 1, 1).cuda()


def report(tag, m, it, loss):
    with torch.no_grad():
        out = m(imgs)[-1] if isinstance(m(imgs), list) else m(imgs)
        loc = out[0, 0].argmax()
        print(
            f"  [{tag}] it{it} loss={loss.item():.4f} peak=({int(loc) % 48},{int(loc) // 48}) value={out[0, 0].max():.2f} (target 24,32)",
            flush=True,
        )


class Tiny(nn.Module):

    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(3, 17, 3, 1, 1)

    def forward(self, x):
        return [
            self.c(
                F.interpolate(x, (OUT_H, OUT_W), mode="bilinear", align_corners=False)
            )
        ]


print("=== A. single conv layer, fixed-peak regression (AdamW 1e-3) ===")
m = Tiny().cuda()
m.apply(_init_from_scratch)
m.train()
opt = torch.optim.AdamW(m.parameters(), lr=0.001)
for it in range(1000):
    opt.zero_grad(set_to_none=True)
    loss = F.mse_loss(m(imgs)[-1], tgt)
    loss.backward()
    opt.step()
    if it in (0, 100, 499, 999):
        report("A", m, it, loss)
print("=== B. P1 + SGD mom0.9 lr0.02 ===")
m, _ = build_model("p1")
m.cuda().train()
opt = torch.optim.SGD(m.parameters(), lr=0.02, momentum=0.9)
for it in range(1000):
    opt.zero_grad(set_to_none=True)
    loss = F.mse_loss(m(imgs)[-1], tgt)
    loss.backward()
    opt.step()
    if it in (0, 100, 499, 999):
        report("B", m, it, loss)
print("=== C. P1 + AdamW 1e-3 (reproduction) ===")
m, _ = build_model("p1")
m.cuda().train()
opt = torch.optim.AdamW(m.parameters(), lr=0.001)
for it in range(1000):
    opt.zero_grad(set_to_none=True)
    loss = F.mse_loss(m(imgs)[-1], tgt)
    loss.backward()
    opt.step()
    if it in (0, 100, 499, 999):
        report("C", m, it, loss)
print("DONE")
