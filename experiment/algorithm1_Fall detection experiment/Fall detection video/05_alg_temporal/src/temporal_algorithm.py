# -*- coding: utf-8 -*-
"""A3: temporal CNN over skeleton windows."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from windowize import make_windows, WIN_FRAMES

IN_DIM = 17 * 3


def _windows_to_tensor(kpts):
    wins, t0s, t1s = make_windows(kpts)
    if len(wins) == 0:
        return None
    # [N,30,17,3] → [N,30,51]
    x = wins.reshape(len(wins), WIN_FRAMES, IN_DIM)
    return torch.from_numpy(x).float()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
class _LSTM(nn.Module):
    def __init__(self, d_in=IN_DIM, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(d_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1]).squeeze(-1)


class _GRU(nn.Module):
    def __init__(self, d_in=IN_DIM, hidden=64):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        o, _ = self.gru(x)
        return self.head(o[:, -1]).squeeze(-1)


class _BiLSTM(nn.Module):
    def __init__(self, d_in=IN_DIM, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(d_in, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1]).squeeze(-1)


class _TCN(nn.Module):

    def __init__(self, d_in=IN_DIM, hidden=64):
        super().__init__()
        self.c1 = nn.Conv1d(d_in, hidden, 5, padding=2)
        self.c2 = nn.Conv1d(hidden, hidden, 5, padding=2)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = F.relu(self.c1(x.transpose(1, 2)))
        h = F.relu(self.c2(h))
        h = h.mean(-1)
        return self.head(h).squeeze(-1)


MODELS = {
    "lstm": _LSTM,
    "gru": _GRU,
    "bilstm": _BiLSTM,
    "tcn": _TCN,
}


class TemporalAlgorithm:

    name = "A3_temporal"

    def __init__(self, epochs=40, batch=128, lr=1e-3, device=None):
        self.epochs, self.batch, self.lr = epochs, batch, lr
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.nets = {}

    def fit(self, train_items):
        Xs, ys = [], []
        for kpts, t0_s, iv in train_items:
            t = _windows_to_tensor(kpts)
            if t is None:
                continue
            labels = np.zeros(len(t), np.float32)
            win_len_s = WIN_FRAMES / 15.0
            for j, w_start in enumerate(t0_s):
                w_end = w_start + win_len_s
                for a, b in iv:
                    if w_start <= b and w_end >= a:
                        labels[j] = 1.0
                        break
            Xs.append(t)
            ys.append(torch.from_numpy(labels).float())
        if not Xs:
            self._const = 0.0
            return self
        X = torch.cat(Xs, 0)
        y = torch.cat(ys, 0)
        if y.sum() == 0 or (y == 1).all():
            self._const = float(y.mean())
            return self
        self._const = None

        self.nets = {}
        for name, cls in MODELS.items():
            net = cls().to(self.device)
            opt = torch.optim.Adam(net.parameters(), lr=self.lr)
            Xd, yd = X.to(self.device), y.to(self.device)
            n = len(Xd)
            for ep in range(self.epochs):
                perm = torch.randperm(n)
                for i in range(0, n, self.batch):
                    idx = perm[i : i + self.batch]
                    opt.zero_grad()
                    logits = net(Xd[idx])
                    loss = F.binary_cross_entropy_with_logits(logits, yd[idx])
                    loss.backward()
                    opt.step()
            self.nets[name] = net
        return self

    def predict(self, kpts):
        t = _windows_to_tensor(kpts)
        if t is None:
            return np.zeros(0, np.float32)
        if self._const is not None:
            return np.full(len(t), self._const, np.float32)
        probs = []
        with torch.no_grad():
            for net in self.nets.values():
                net.eval()
                probs.append(torch.sigmoid(net(t.to(self.device))).cpu().numpy())
        return np.mean(probs, axis=0).astype(np.float32)
