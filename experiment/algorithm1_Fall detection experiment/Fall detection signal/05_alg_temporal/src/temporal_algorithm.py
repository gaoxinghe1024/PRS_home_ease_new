# -*- coding: utf-8 -*-
"""A3 temporal deep networks (raw acceleration windows -> end-to-end temporal modelling).

Idea: skip handcrafted features and feed the multi-channel 2 s acceleration
sequence [120,4] (ax, ay, az, |a|, standardised with training statistics) straight
into a temporal network, letting it learn the "free fall -> impact -> rest" dynamics.

Internal architecture comparison (single variable: same data/hyperparameters, only the
architecture differs):
  LSTM   unidirectional long short-term memory
  GRU    gated recurrent unit (fewer parameters, more stable on small samples)
  BiLSTM bidirectional LSTM (sees both pre-impact and post-impact context)
  TCN    temporal convolution (dilated causal convolution + residual; controllable
         receptive field, fully parallel)

Final output = soft-voting average of the four window probability vectors.

Small-sample measures: hidden=32 single layer, dropout 0.15, weight decay 1e-4,
positive-weighted BCEWithLogitsLoss (only ~17% positive windows).

Interface matches A1/A2/A4: fit(train) / predict(wins).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "00_common"))

import torch
import torch.nn as nn
import torch.nn.functional as Fn

IN_DIM = 4
MODEL_NAMES = ("lstm", "gru", "bilstm", "tcn")


# ---------------------------------------------------------------------------
# Network definitions
# ---------------------------------------------------------------------------
class _LSTM(nn.Module):
    def __init__(self, d_in=IN_DIM, hidden=32, bidir=False, drop=0.15):
        super().__init__()
        self.rnn = nn.LSTM(d_in, hidden, batch_first=True, bidirectional=bidir)
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(hidden * (2 if bidir else 1), 1)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.head(self.drop(o[:, -1])).squeeze(-1)


class _GRU(nn.Module):
    def __init__(self, d_in=IN_DIM, hidden=32, drop=0.15):
        super().__init__()
        self.rnn = nn.GRU(d_in, hidden, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.head(self.drop(o[:, -1])).squeeze(-1)


class _BiLSTM(nn.Module):
    def __init__(self, d_in=IN_DIM, hidden=32, drop=0.15):
        super().__init__()
        self.net = _LSTM(d_in, hidden, bidir=True, drop=drop)

    def forward(self, x):
        return self.net(x)


class _TCN(nn.Module):
    """Dilated causal convolution + residual (temporal convolutional network)."""

    def __init__(self, d_in=IN_DIM, ch=32, levels=(1, 2, 4, 8), k=3, drop=0.15):
        super().__init__()
        self.inp = nn.Conv1d(d_in, ch, 1)
        self.blocks = nn.ModuleList()
        for d in levels:
            self.blocks.append(nn.Sequential(
                nn.Conv1d(ch, ch, k, dilation=d, padding=d * (k - 1)),
                nn.BatchNorm1d(ch), nn.ReLU(), nn.Dropout(drop),
                nn.Conv1d(ch, ch, k, dilation=d, padding=d * (k - 1)),
                nn.BatchNorm1d(ch), nn.ReLU(),
            ))
        self.head = nn.Linear(ch, 1)

    def forward(self, x):
        h = self.inp(x.transpose(1, 2))          # [B,C,T]
        for blk in self.blocks:
            out = blk(h)
            h = Fn.relu(h + out[:, :, :h.shape[-1]])
        h = h.mean(dim=-1)                        # global average pooling
        return self.head(h).squeeze(-1)


BUILDERS = {
    "lstm": lambda: _LSTM(),
    "gru": lambda: _GRU(),
    "bilstm": lambda: _BiLSTM(),
    "tcn": lambda: _TCN(),
}


# ---------------------------------------------------------------------------
# Training / inference wrapper
# ---------------------------------------------------------------------------
class _TorchModel:
    def __init__(self, key, seed=0, epochs=25, lr=2e-3, batch=64, device="cpu"):
        self.key = key
        self.seed = seed
        self.epochs = epochs
        self.lr = lr
        self.batch = batch
        self.device = device
        self.net = None
        self.mean = None
        self.std = None

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X = np.asarray(X, np.float32)
        y = np.asarray(y, np.float32)
        self.mean = X.reshape(-1, X.shape[-1]).mean(0)
        self.std = X.reshape(-1, X.shape[-1]).std(0) + 1e-6
        Xn = (X - self.mean) / self.std
        xt = torch.from_numpy(Xn)
        yt = torch.from_numpy(y)
        self.net = BUILDERS[self.key]().to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        pos_w = torch.tensor([max((y == 0).sum() / max((y == 1).sum(), 1.0), 1.0)],
                             dtype=torch.float32, device=self.device)
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
        n = len(xt)
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n)
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                if len(idx) < 2:
                    continue
                opt.zero_grad()
                out = self.net(xt[idx])
                loss = lossf(out, yt[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 3.0)
                opt.step()
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        if self.net is None or len(X) == 0:
            return np.zeros(len(X), np.float64)
        self.net.eval()
        Xn = (np.asarray(X, np.float32) - self.mean) / self.std
        out = self.net(torch.from_numpy(Xn.astype(np.float32)))
        return torch.sigmoid(out).numpy().astype(np.float64)


class TemporalAlgorithm:
    """A3: soft voting over four temporal networks."""

    name = "A3_temporal_nets"

    def __init__(self, fs=60.0, seed=0, nets=MODEL_NAMES, epochs=25, lr=2e-3):
        self.fs = fs
        self.seed = seed
        self.nets = tuple(nets)
        self.epochs = epochs
        self.lr = lr
        self.models = {}
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    def fit(self, train):
        self.fs = train.get("fs", self.fs)
        X = np.asarray(train["wins"], np.float32)
        y = np.asarray(train["labels"], np.float32)
        self.models = {}
        for i, k in enumerate(self.nets):
            m = _TorchModel(k, seed=self.seed + 17 * i, epochs=self.epochs, lr=self.lr)
            m.fit(X, y)
            self.models[k] = m
        return self

    def predict(self, wins):
        if len(self.models) == 0 or len(wins) == 0:
            return np.zeros(len(wins), np.float64)
        X = np.asarray(wins, np.float32)
        probs = [m.predict_proba(X) for m in self.models.values()]
        return np.mean(probs, axis=0)

    def predict_per_net(self, wins):
        X = np.asarray(wins, np.float32)
        return {k: m.predict_proba(X) for k, m in self.models.items()}
