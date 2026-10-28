"""Export pose weights and render demo predictions."""

import os
import sys
import json
import argparse
import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from dataset_coco import affine_crop, IMG_H, IMG_W, NUM_KPTS, COCO_EDGES
from models import build_model

DEFAULT_CKPT = os.path.join(HERE, "outputs", "best.pt")


class PoseCNNExtractor:

    def __init__(self, ckpt_path=DEFAULT_CKPT, device=None):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"weights not found {ckpt_path}; run train.py / exp_arch_compare first"
            )
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.arch = ck.get("arch", "p1")
        self.model, _ = build_model(self.arch)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.val_pck02 = ck.get("val_pck02")
        self._expand = 1.25

    def extract(self, img, bbox_px):
        H, W = img.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox_px]
        bbox = np.asarray(
            [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)], dtype=np.float32
        )
        crop, M = affine_crop(img, bbox, IMG_H, IMG_W, expand=self._expand)
        Minv = cv2.invertAffineTransform(M)
        t = torch.from_numpy(np.ascontiguousarray(crop[:, :, ::-1]))
        t = t.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            hm = self.model(t.to(self.device))[-1]
        hm = hm[0]
        conf, idx = hm.view(NUM_KPTS, -1).max(-1)
        xs = (idx % hm.shape[2]).float() * (IMG_W / hm.shape[2])
        ys = (idx // hm.shape[2]).float() * (IMG_H / hm.shape[1])
        pts_in = torch.stack([xs, ys], -1).cpu().numpy().astype(np.float64)
        pts_orig = pts_in @ Minv[:, :2].T + Minv[:, 2]
        return (pts_orig.astype(np.float32), conf.cpu().numpy())

    def extract_person(self, img, bbox_norm_xyxy):
        H, W = img.shape[:2]
        bb = [
            bbox_norm_xyxy[0] * W,
            bbox_norm_xyxy[1] * H,
            bbox_norm_xyxy[2] * W,
            bbox_norm_xyxy[3] * H,
        ]
        return self.extract(img, bb)


def run_demo(n=12):
    ann_path = os.path.join(
        ROOT, "data", "coco", "annotations", "person_keypoints_val2017.json"
    )
    img_root = os.path.join(ROOT, "data", "coco", "val2017")
    out_dir = os.path.join(HERE, "outputs", "demo")
    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, HERE)
    from dataset_coco import load_coco_person_annots

    anns = load_coco_person_annots(ann_path)
    ext = PoseCNNExtractor()
    print(f"[demo] arch={ext.arch} val_pck02={ext.val_pck02}")
    for i in range(n):
        a = anns[i]
        img = cv2.imread(os.path.join(img_root, a["file_name"]))
        bx, by, bw, bh = a["bbox"]
        kpts, conf = ext.extract(img, [bx, by, bx + bw, by + bh])
        for i0, i1 in COCO_EDGES:
            p0, p1 = (kpts[i0], kpts[i1])
            if conf[i0] > 0.2 and conf[i1] > 0.2:
                cv2.line(
                    img, tuple(p0.astype(int)), tuple(p1.astype(int)), (0, 255, 0), 2
                )
        for j in range(NUM_KPTS):
            c = int(255 * conf[j])
            cv2.circle(img, tuple(kpts[j].astype(int)), 3, (c, 0, 255 - c), -1)
        cv2.rectangle(
            img, (int(bx), int(by)), (int(bx + bw), int(by + bh)), (255, 128, 0), 1
        )
        path = os.path.join(out_dir, f"demo_{i:02d}.jpg")
        cv2.imwrite(path, img)
    print(f"[demo] {n} visualizations -> outputs/demo/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--demo", type=int, default=0, help="number of visualization samples"
    )
    a = ap.parse_args()
    if a.demo:
        run_demo(a.demo)
    else:
        print("usage: python export_infer.py --demo 12")
