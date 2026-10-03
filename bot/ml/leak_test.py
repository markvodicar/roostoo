"""Mechanical look-ahead checks. Run: python -m bot.ml.leak_test

1. Feature causality: features computed on data truncated at hour T must equal the
   features at T computed on the full history. If any future bar leaked into a
   feature, the two would differ.
2. Target alignment: target at T must equal log(close[T+h]) - log(close[T]) / vol[T].
3. Sample windows: a training window ending at T only indexes rows <= T.
4. Split embargo: no training target extends past the validation start, etc.
5. Simulation timing: weights set at hour T earn only returns from T -> T+1 onward
   (checked by feeding a synthetic "oracle" prediction = next-hour return and a
   lagged one; the lagged version must NOT be able to profit from it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from bot.ml.dataset import EMBARGO_H, SPLITS, WindowDataset, get_panel, sample_index
from bot.ml.features import HORIZONS, _asset_frame, build_panel, load_ohlcv, targets
from bot.ml.model import WINDOW
from bot.ml.portfolio import PortParams, simulate


def check_feature_causality():
    ohlcv = load_ohlcv(["BTC", "ETH", "SOL", "DOGE"])
    for c, df in ohlcv.items():
        full = _asset_frame(df)
        for T in (len(df) // 3, len(df) // 2, len(df) - 1000):
            trunc = _asset_frame(df.iloc[: T + 1])
            a, b = full.iloc[T].fillna(0).values, trunc.iloc[-1].fillna(0).values
            assert np.allclose(a, b, atol=1e-9, equal_nan=True), f"{c} feature mismatch at {T}: {a} vs {b}"
    # cross-asset channels (btc, mkt) on the panel
    full = build_panel(ohlcv)
    T = len(full[4]) - 500
    cut = {c: d[d.index <= full[4][T]] for c, d in ohlcv.items()}
    part = build_panel(cut)
    assert np.allclose(full[0][T], part[0][-1], atol=1e-6), "panel features differ when future removed"
    print("1. feature causality: OK")


def check_targets():
    feat, vol, lc, valid, times, coins = get_panel()
    y = targets(lc, vol)
    j = coins.index("BTC")
    for k, h in enumerate(HORIZONS):
        T = 50_000
        expect = np.clip((lc[T + h, j] - lc[T, j]) / (vol[T, j] * np.sqrt(h)), -4, 4)
        assert np.isclose(y[T, j, k], expect), (y[T, j, k], expect)
        assert np.isnan(y[-1, j, k]), "last target must be NaN (no future)"
    print("2. target alignment: OK")
    return feat, vol, lc, valid, times, coins, y


def check_windows_and_embargo(feat, valid, times, y):
    idx = sample_index(valid, times, "train", y)
    ds = WindowDataset(feat, y, idx)
    t, j = idx[-1]
    x, yy = ds.gather(torch.tensor([len(ds) - 1]))
    x = x[0]
    assert np.array_equal(x.numpy(), feat[t - WINDOW + 1: t + 1, j]), "window indexes wrong rows"
    assert t - WINDOW + 1 >= 0
    print("3. sample windows: OK")
    for s, nxt in (("train", "val"), ("val", "test"), ("test", "holdout")):
        last_t = times[sample_index(valid, times, s, y)[:, 0].max()]
        nxt_start = pd.Timestamp(SPLITS[nxt][0], tz="UTC")
        assert last_t + pd.Timedelta(hours=max(HORIZONS)) < nxt_start, f"{s} targets reach into {nxt}"
        assert EMBARGO_H >= max(HORIZONS)
    print("4. split embargo: OK")


def check_simulation_timing(vol, lc, times, coins):
    t0 = times.get_loc(pd.Timestamp("2024-01-01", tz="UTC"))
    sl = slice(t0, t0 + 24 * 60)
    lcf = pd.DataFrame(lc[sl], index=times[sl], columns=coins).dropna(axis=1)
    volf = pd.DataFrame(vol[sl], index=times[sl], columns=coins)[lcf.columns]
    nxt = lcf.shift(-1) - lcf  # return from T to T+1, known only at T+1
    p = PortParams(top_k=3, rebal_h=1, band=0.0, target_vol=5.0, max_w=1.0)
    eq_oracle, _ = simulate(nxt, volf, lcf, p, fee=0.0)
    eq_lagged, _ = simulate(nxt.shift(1), volf, lcf, p, fee=0.0)  # knows only up to T-1 -> T
    r_o, r_l = eq_oracle.iloc[-1] - 1, eq_lagged.iloc[-1] - 1
    assert r_o > 1.0, f"oracle should make a fortune, got {r_o:.2f}"
    assert r_l < r_o / 10, f"lagged oracle profits too much ({r_l:.2f}) -> simulator leaks a bar"
    print(f"5. simulation timing: OK (oracle {r_o:+.1%}, 1h-lagged oracle {r_l:+.1%})")


if __name__ == "__main__":
    check_feature_causality()
    feat, vol, lc, valid, times, coins, y = check_targets()
    check_windows_and_embargo(feat, valid, times, y)
    check_simulation_timing(vol, lc, times, coins)
    print("\nAll leakage checks passed.")
