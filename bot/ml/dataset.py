"""Build the pooled (time, asset) sample index and the 6/1/3 split, cache tensors."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bot.ml.features import build_panel, load_ohlcv, targets
from bot.ml.model import WINDOW

CACHE = Path(__file__).resolve().parents[2] / "data" / "panel.npz"
SPLITS = {  # inclusive start, exclusive end
    "train": ("2016-01-01", "2022-01-01"),
    "val": ("2022-01-01", "2023-01-01"),
    "test": ("2023-01-01", "2026-01-01"),
    "holdout": ("2026-01-01", "2100-01-01"),
}
EMBARGO_H = 336  # drop samples whose 14-day target overlaps the next split


def get_panel(refresh: bool = False):
    if CACHE.exists() and not refresh:
        z = np.load(CACHE, allow_pickle=True)
        times = pd.DatetimeIndex(z["times"])
        times = times.tz_localize("UTC") if times.tz is None else times
        return (z["feat"], z["vol"], z["lc"], z["valid"], times, list(z["coins"]))
    feat, vol, lc, valid, times, coins = build_panel(load_ohlcv())
    np.savez_compressed(CACHE, feat=feat, vol=vol, lc=lc, valid=valid,
                        times=times.values, coins=np.array(coins))
    return feat, vol, lc, valid, times, coins


def sample_index(valid: np.ndarray, times: pd.DatetimeIndex, split: str, y: np.ndarray | None = None) -> np.ndarray:
    """[(t, j)] pairs with a full window of history and (for training) a target."""
    s, e = (pd.Timestamp(x, tz="UTC") for x in SPLITS[split])
    tmask = (times >= s) & (times < e)
    if split != "holdout":
        tmask &= times < e - pd.Timedelta(hours=EMBARGO_H)
    tmask[:WINDOW] = False
    ok = valid.copy()
    # whole window must be valid (asset listed throughout)
    win_ok = np.ones_like(valid)
    cs = np.cumsum(np.concatenate([np.zeros((1, valid.shape[1]), int), valid.astype(int)]), 0)
    win_ok[WINDOW:] = (cs[WINDOW + 1:] - cs[1:-WINDOW]) == WINDOW
    win_ok[:WINDOW] = False
    ok &= win_ok & tmask[:, None]
    if y is not None:
        ok &= ~np.isnan(y).any(-1)
    return np.argwhere(ok)


class WindowDataset:
    """Vectorised batch gatherer: x[b] = feat[t_b-W+1 : t_b+1, j_b, :]."""

    def __init__(self, feat: np.ndarray, y: np.ndarray | None, idx: np.ndarray):
        self.feat = torch.from_numpy(feat)
        self.y = None if y is None else torch.from_numpy(y)
        self.idx = torch.from_numpy(idx.astype(np.int64))
        self.offs = torch.arange(-WINDOW + 1, 1)

    def __len__(self):
        return len(self.idx)

    def gather(self, rows: torch.Tensor):
        t, j = self.idx[rows, 0], self.idx[rows, 1]
        x = self.feat[(t[:, None] + self.offs), j[:, None], :]          # [B, W, C]
        if self.y is None:
            return x
        return x, self.y[t, j]

    def batches(self, bs: int, shuffle: bool = False, n_samples: int | None = None):
        n = len(self)
        order = torch.randperm(n) if shuffle else torch.arange(n)
        if n_samples is not None:
            order = order[:n_samples]
        for i in range(0, len(order), bs):
            yield self.gather(order[i: i + bs])


if __name__ == "__main__":
    feat, vol, lc, valid, times, coins = get_panel(refresh=True)
    y = targets(lc, vol)
    print(f"panel {feat.shape}, {times[0]} -> {times[-1]}, {len(coins)} coins")
    for s in SPLITS:
        n = len(sample_index(valid, times, s, y if s != "holdout" else None))
        print(f"  {s:8s} {n:>9,d} samples")
