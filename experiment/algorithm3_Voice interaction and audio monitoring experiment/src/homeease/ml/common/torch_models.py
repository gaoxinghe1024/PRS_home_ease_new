"""Causal temporal models: unidirectional LSTM and dilated causal TCN.

Both read a window (B, T, C) and predict from the LAST time step, so the output
for a window ending at t never uses samples after t. ``per_step`` outputs are
available for causality tests and streaming use. Training runs on CPU with fixed
seeds for reproducibility.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any

import numpy as np
import torch
from torch import nn


class UniLSTM(nn.Module):
    def __init__(self, c_in: int, hidden: int = 64, layers: int = 1, n_out: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.cfg = {"arch": "lstm", "c_in": c_in, "hidden": hidden, "layers": layers, "n_out": n_out, "dropout": dropout}
        self.lstm = nn.LSTM(c_in, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0, bidirectional=False)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_out)

    def per_step(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        return self.head(self.drop(h))  # (B, T, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.per_step(x)[:, -1]
        return out.squeeze(-1) if out.shape[-1] == 1 else out


class _CausalBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int, d: int, dropout: float) -> None:
        super().__init__()
        self.pad = (k - 1) * d
        self.c1 = nn.Conv1d(c_in, c_out, k, dilation=d)
        self.c2 = nn.Conv1d(c_out, c_out, k, dilation=d)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T)
        y = self.c1(nn.functional.pad(x, (self.pad, 0)))
        y = self.drop(torch.relu(y))
        y = self.c2(nn.functional.pad(y, (self.pad, 0)))
        y = self.drop(torch.relu(y))
        return torch.relu(y + self.res(x))


class CausalTCN(nn.Module):
    def __init__(self, c_in: int, channels: int = 32, kernel: int = 3, dilations: tuple[int, ...] = (1, 2, 4, 8, 16), n_out: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.cfg = {"arch": "tcn", "c_in": c_in, "channels": channels, "kernel": kernel, "dilations": list(dilations), "n_out": n_out, "dropout": dropout}
        blocks = []
        c = c_in
        for d in dilations:
            blocks.append(_CausalBlock(c, channels, kernel, d, dropout))
            c = channels
        self.net = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, n_out)

    @property
    def receptive_field(self) -> int:
        k = self.cfg["kernel"]
        return 1 + 2 * (k - 1) * sum(self.cfg["dilations"])

    def per_step(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.head(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.per_step(x)[:, -1]
        return out.squeeze(-1) if out.shape[-1] == 1 else out


def build(cfg: dict[str, Any]) -> nn.Module:
    c = dict(cfg)
    arch = c.pop("arch")
    if arch == "lstm":
        return UniLSTM(**c)
    if arch == "tcn":
        c["dilations"] = tuple(c["dilations"])
        return CausalTCN(**c)
    raise ValueError(arch)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def _metric(y: np.ndarray, s: np.ndarray, multiclass: bool) -> float:
    if multiclass:
        from sklearn.metrics import f1_score

        return float(f1_score(y, s.argmax(1), average="macro"))
    from sklearn.metrics import average_precision_score

    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(average_precision_score(y, s))


def predict(model: nn.Module, X: np.ndarray, batch: int = 512, multiclass: bool = False) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i : i + batch].astype(np.float32))
            o = model(xb)
            outs.append(torch.softmax(o, -1).numpy() if multiclass else torch.sigmoid(o).numpy())
    return np.concatenate(outs) if outs else np.zeros((0,))


def train(
    model: nn.Module,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    *,
    epochs: int = 40,
    lr: float = 1e-3,
    batch: int = 64,
    seed: int = 0,
    patience: int = 6,
    multiclass: bool = False,
    augment: Any = None,
    class_weights: np.ndarray | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    if multiclass:
        w = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        loss_fn: Any = nn.CrossEntropyLoss(weight=w)
    else:
        pos = float(ytr.sum())
        neg = float(len(ytr) - pos)
        pw = torch.tensor(min(neg / max(pos, 1.0), 50.0))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    rng = np.random.default_rng(seed)
    best, best_state, best_epoch, bad = -math.inf, None, -1, 0
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        idx = rng.permutation(len(Xtr))
        tot = 0.0
        for i in range(0, len(idx), batch):
            b = idx[i : i + batch]
            xb = Xtr[b]
            if augment is not None:
                xb = augment(xb, rng)
            xt = torch.from_numpy(xb.astype(np.float32))
            yt = torch.from_numpy(ytr[b].astype(np.int64 if multiclass else np.float32))
            opt.zero_grad()
            loss = loss_fn(model(xt), yt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(b)
        sva = predict(model, Xva, multiclass=multiclass)
        m = _metric(yva, sva, multiclass) if len(Xva) else float("nan")
        hist.append({"epoch": ep, "train_loss": tot / max(len(idx), 1), "val_metric": m})
        score = m if np.isfinite(m) else -tot
        if score > best + 1e-4:
            best, best_state, best_epoch, bad = score, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"history": hist, "best_epoch": best_epoch, "best_val_metric": best, "val_metric_name": "macro_f1" if multiclass else "average_precision", "seconds": round(time.time() - t0, 1)}
