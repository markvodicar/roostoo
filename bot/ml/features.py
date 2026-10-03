"""Feature engineering shared by training, backtest and live inference.

Every feature at hour t uses only data up to and including the close of hour t.
All features are scale-free (vol-normalised) so one network can be pooled across
coins with prices from $0.00001 to $100k.

Channels (per asset, per hour):
  0 r1     1h log return / rolling vol
  1 hl     log(high/low) / rolling vol          (intrabar range)
  2 vlm    log dollar volume vs 7d mean          (activity shock)
  3 dev7   log(close / EMA 7d)  / (vol*sqrt(168))
  4 dev30  log(close / EMA 30d) / (vol*sqrt(720))
  5 lvol   log annualised vol (centred)          (vol regime)
  6 btc    BTC r1 / BTC vol                      (market leader)
  7 mkt    cross-sectional mean of r1            (market breadth/momentum)
Targets: forward log return over H hours / (vol*sqrt(H)), clipped.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data" / "full"
VOL_WIN = 168
HORIZONS = (24, 72, 336)  # 1 day, 3 days, 14 days (= competition window)
N_CH = 8
MIN_BARS = VOL_WIN * 2   # skip the first 2 weeks of each listing (vol estimate unreliable)
EXCLUDE = {"PAXG"}       # gold-backed token: different asset class


def load_ohlcv(coins: list[str] | None = None) -> dict[str, pd.DataFrame]:
    out = {}
    for f in sorted(DATA.glob("*.csv")):
        c = f.stem
        if c in EXCLUDE or (coins is not None and c not in coins):
            continue
        out[c] = regularize(pd.read_csv(f, index_col=0, parse_dates=True))
    return out


def regularize(df: pd.DataFrame) -> pd.DataFrame:
    """Regular hourly grid; short gaps forward-filled (close), volume 0. Used by both
    the training loader and the live bot so features are computed identically."""
    df = df[~df.index.duplicated()].sort_index()
    idx = pd.date_range(df.index[0], df.index[-1], freq="1h")
    df = df.reindex(idx)
    df["volume_usd"] = df["volume_usd"].fillna(0)
    df[["close"]] = df[["close"]].ffill(limit=6)
    for col in ("open", "high", "low"):
        df[col] = df[col].fillna(df["close"])
    return df.dropna(subset=["close"])


def _asset_frame(df: pd.DataFrame) -> pd.DataFrame:
    lc = np.log(df["close"])
    r1 = lc.diff()
    vol = r1.rolling(VOL_WIN, min_periods=VOL_WIN // 2).std().clip(lower=1e-4)
    f = pd.DataFrame(index=df.index)
    f["r1"] = (r1 / vol).clip(-6, 6)
    f["hl"] = (np.log(df["high"] / df["low"]) / vol).clip(0, 10)
    lv = np.log1p(df["volume_usd"])
    f["vlm"] = (lv - lv.rolling(VOL_WIN, min_periods=24).mean()).clip(-5, 5)
    for name, span in (("dev7", 168), ("dev30", 720)):
        ema = lc.ewm(span=span, adjust=False, min_periods=span // 4).mean()
        f[name] = ((lc - ema) / (vol * np.sqrt(span))).clip(-5, 5)
    f["lvol"] = np.log(vol * np.sqrt(24 * 365)) + 0.5   # ~0 at 60% annual vol
    f["_vol"] = vol
    f["_lc"] = lc
    return f


def build_panel(ohlcv: dict[str, pd.DataFrame]):
    """Returns (feat [T, N, C] float32, vol [T, N], logclose [T, N], valid [T, N] bool, times, coins)."""
    frames = {c: _asset_frame(df) for c, df in ohlcv.items()}
    times = pd.DatetimeIndex(sorted(set().union(*[f.index for f in frames.values()])))
    coins = sorted(frames)
    T, N = len(times), len(coins)
    feat = np.zeros((T, N, N_CH), np.float32)
    vol = np.full((T, N), np.nan, np.float32)
    lc = np.full((T, N), np.nan, np.float64)
    valid = np.zeros((T, N), bool)
    for j, c in enumerate(coins):
        f = frames[c].reindex(times)
        feat[:, j, :6] = f[["r1", "hl", "vlm", "dev7", "dev30", "lvol"]].fillna(0).values
        vol[:, j] = f["_vol"].values
        lc[:, j] = f["_lc"].values
        n_seen = f["_lc"].notna().cumsum().values
        valid[:, j] = f["_lc"].notna().values & f["_vol"].notna().values & (n_seen >= MIN_BARS)
    r1 = feat[:, :, 0]
    if "BTC" in coins:
        feat[:, :, 6] = r1[:, coins.index("BTC")][:, None]
    live = ~np.isnan(lc)
    mkt = np.where(live, r1, 0).sum(1) / np.maximum(live.sum(1), 1)
    feat[:, :, 7] = mkt[:, None]
    return feat, vol, lc, valid, times, coins


def targets(lc: np.ndarray, vol: np.ndarray, horizons=HORIZONS) -> np.ndarray:
    """[T, N, H] vol-normalised forward log returns (NaN where unavailable)."""
    T, N = lc.shape
    out = np.full((T, N, len(horizons)), np.nan, np.float32)
    for k, h in enumerate(horizons):
        fwd = np.full_like(lc, np.nan)
        fwd[:-h] = lc[h:] - lc[:-h]
        out[:, :, k] = np.clip(fwd / (vol * np.sqrt(h)), -4, 4)
    return out


def raw_forward(lc: np.ndarray, h: int) -> np.ndarray:
    fwd = np.full_like(lc, np.nan)
    fwd[:-h] = lc[h:] - lc[:-h]
    return fwd
