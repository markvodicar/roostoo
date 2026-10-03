"""1D CNN over a window of hourly features for one asset -> predicted
vol-normalised forward returns at several horizons (multi-task regression).

Architecture (small on purpose: ~60k params, 10 years of hourly data is still
only ~3M noisy samples):
  Conv1d stack with dilations 1,2,4,8 (receptive field ~ 2 x window) + GELU +
  LayerNorm, global average + last-step pooling, MLP head.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from bot.ml.features import HORIZONS, N_CH

WINDOW = 96  # hours of context


class ResBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, k: int = 5, drop: float = 0.1):
        super().__init__()
        pad = (k - 1) * dilation  # causal padding
        self.pad = pad
        self.conv1 = nn.Conv1d(ch, ch, k, dilation=dilation)
        self.conv2 = nn.Conv1d(ch, ch, 1)
        self.norm = nn.GroupNorm(1, ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        h = nn.functional.pad(x, (self.pad, 0))
        h = self.act(self.conv1(h))
        h = self.drop(self.conv2(h))
        return self.norm(x + h)


class CNN(nn.Module):
    def __init__(self, n_ch: int = N_CH, hidden: int = 64, n_out: int = len(HORIZONS), drop: float = 0.1):
        super().__init__()
        self.inp = nn.Conv1d(n_ch, hidden, 1)
        self.blocks = nn.Sequential(*[ResBlock(hidden, d, drop=drop) for d in (1, 2, 4, 8, 16)])
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, n_out))

    def forward(self, x):  # x: [B, W, C]
        h = self.inp(x.transpose(1, 2))
        h = self.blocks(h)
        pooled = torch.cat([h.mean(-1), h[:, :, -1]], dim=1)
        return self.head(pooled)
