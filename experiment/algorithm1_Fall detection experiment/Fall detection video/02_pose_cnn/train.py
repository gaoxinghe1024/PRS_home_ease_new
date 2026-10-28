"""Single-architecture pose training loop."""

import os
import sys
import json
import time
import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from dataset_coco import load_coco_person_annots, CocoPoseDataset
from models import build_model, count_params
from eval_pose import pck_on_loader

COCO_DIR = os.path.join(ROOT, "data", "coco")


def build_loaders(batch_size, workers=2, val_max=None, use_train2017=True):
    ann_dir = os.path.join(COCO_DIR, "annotations")
    train_ann = os.path.join(ann_dir, "person_keypoints_train2017.json")
    val_ann = os.path.join(ann_dir, "person_keypoints_val2017.json")
    if use_train2017 and os.path.exists(train_ann):
        train_dir = os.path.join(COCO_DIR, "train2017")
        n_train_img = len(os.listdir(train_dir)) if os.path.isdir(train_dir) else 0
        if n_train_img >= 100000:
            train_samples = load_coco_person_annots(train_ann)
            val_samples = load_coco_person_annots(val_ann)
            train_root = os.path.join(COCO_DIR, "train2017")
            val_root = os.path.join(COCO_DIR, "val2017")
            src = "train2017"
        else:
            train_samples = val_samples = None
            src = f"train2017 not ready ({n_train_img} images)"
    else:
        train_samples = val_samples = None
        src = "val2017 (fallback split)"
    if train_samples is None:
        val_samples = load_coco_person_annots(val_ann)
        rng = np.random.RandomState(2026)
        idx = rng.permutation(len(val_samples))
        n_train = int(len(idx) * 0.9)
        train_samples = [val_samples[i] for i in idx[:n_train]]
        val_samples = [val_samples[i] for i in idx[n_train:]]
        train_root = val_root = os.path.join(COCO_DIR, "val2017")
        src += " -> fallback split"
    print(f"[data] train={len(train_samples)} val={len(val_samples)} ({src})")
    if val_max:
        val_samples = val_samples[:val_max]
    train_ds = CocoPoseDataset(train_root, train_samples, training=True)
    val_ds = CocoPoseDataset(val_root, val_samples, training=False)
    train_ld = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=True,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    return (train_ld, val_ld)


def heatmap_loss(pred_hms, gt_hm, vis, pos_weight=50.0):
    tgt = gt_hm * vis[:, :, None, None]
    w = 1.0 + (pos_weight - 1.0) * (tgt > 0.5).float()
    losses = []
    for h in pred_hms:
        se = F.mse_loss(h, tgt, reduction="none") * w
        losses.append(se.mean())
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["p1", "p2", "p3"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--wd", type=float, default=0.0001)
    ap.add_argument(
        "--pos-weight",
        type=float,
        default=50.0,
        help="loss weight for positive heatmap cells (breaks the all-zero saddle point; default 50)",
    )
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument(
        "--val-max",
        type=int,
        default=None,
        help="truncate the validation set (quick debugging)",
    )
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[env] device={device} ({(torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu')})"
    )
    model, n_stages = build_model(a.arch)
    model.to(device)
    print(
        f"[model] {a.arch} params={count_params(model) / 1000000.0:.2f}M stages={n_stages}"
    )
    train_ld, val_ld = build_loaders(a.batch, a.workers, a.val_max)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    total_steps = len(train_ld) * a.epochs
    warmup_steps = len(train_ld) * a.warmup

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    out_dir = os.path.join(HERE, "outputs", a.arch)
    os.makedirs(out_dir, exist_ok=True)
    history = []
    best_pck = -1.0
    for ep in range(1, a.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss, nb = (0.0, 0)
        for batch in train_ld:
            imgs = batch["img"].to(device, non_blocking=True)
            hm = batch["heatmap"].to(device, non_blocking=True)
            vis = batch["vis"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                pred = model(imgs)
                loss = heatmap_loss(pred, hm, vis, a.pos_weight)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += float(loss.item())
            nb += 1
        val = pck_on_loader(model, val_ld, device)
        rec = {
            "epoch": ep,
            "train_loss": run_loss / max(1, nb),
            "val_pck02": val["pck@0.2"],
            "val_pck05": val["pck@0.5"],
            "lr": sched.get_last_lr()[0],
            "sec": round(time.time() - t0, 1),
        }
        history.append(rec)
        print(
            f"[{a.arch}] ep{ep:02d}/{a.epochs} loss={rec['train_loss']:.4f} val PCK@0.2={rec['val_pck02']:.4f} PCK@0.5={rec['val_pck05']:.4f} lr={rec['lr']:.2e} {rec['sec']}s",
            flush=True,
        )
        with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"arch": a.arch, "config": vars(a), "history": history},
                f,
                ensure_ascii=False,
                indent=1,
            )
        if val["pck@0.2"] > best_pck:
            best_pck = val["pck@0.2"]
            torch.save(
                {
                    "arch": a.arch,
                    "state_dict": model.state_dict(),
                    "epoch": ep,
                    "val_pck02": best_pck,
                    "config": vars(a),
                },
                os.path.join(out_dir, "best.pt"),
            )
    print(
        f"[{a.arch}] training done, best val PCK@0.2={best_pck:.4f} → outputs/{a.arch}/best.pt"
    )


if __name__ == "__main__":
    main()
