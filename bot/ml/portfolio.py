"""Turn model predictions into a portfolio and simulate it with Roostoo fees.

Signal -> weights (identical code used live):
  * keep assets with a full feature window; score = predicted 24h vol-normalised return
  * long book: top_k by score among score > thresh; optional short book: bottom_k
    (Roostoo shorts cost 0.1% open + 0.1% close, no leverage; collateral in USD)
  * inverse-vol sizing, scaled to a portfolio vol target, capped per name
  * rebalance every `rebal_h` hours; only trade drifts above `band`
  * drawdown throttle (bot/risk.py) scales gross exposure

Simulation marks positions hourly at close and charges fees on traded notional.
Fee schedule comes from bot/fees.py (competition rule 8: 0.10% taker, 0.05% maker).

Fill model (simulate kwargs, defaults reproduce the plain all-taker simulation):
  * maker_frac   fraction of each rebalance's traded notional assumed to fill as a
                 resting limit order at `maker_fee` (default MAKER); the remainder
                 crosses the book and pays `fee` (default TAKER).
  * spread_bps   half-spread cost in bps per unit of *taker* traded notional
                 (maker fills rest at the touch and do not cross the spread).
                 Scalar, or a Series/dict indexed by asset; assets missing from the
                 Series get its median. Measured Roostoo quoted spreads (2026-10):
                 median ~2.8 bps, meme coins 15-28 bps, BTC/ETH ~0.
  * Short-side notional always pays at least SHORT_OPEN/SHORT_CLOSE (0.10%).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bot.backtest import metrics, window_report
from bot.fees import MAKER, SHORT_CLOSE, SHORT_OPEN, TAKER
from bot.ml.windows import summarize
from bot.risk import RiskParams, RollingPeak, exposure_scale

SHORT_FEE = max(SHORT_OPEN, SHORT_CLOSE)  # floor rate on any notional touching a short position
MEDIAN_SPREAD_BPS = 2.8                   # measured median Roostoo quoted spread (bps), conservative half-spread proxy
HPY = 24 * 365


@dataclass
class PortParams:
    top_k: int = 5
    short_k: int = 0
    thresh: float = 0.0          # min predicted score to go long (in vol units)
    short_thresh: float = 0.0    # max predicted score to go short
    target_vol: float = 0.50
    max_w: float = 0.35
    rebal_h: int = 24
    band: float = 0.05
    gross_cap: float = 1.0
    use_regime: bool = False     # also require mkt feature (breadth) > 0 to be long
    head: int = 336              # which forecast horizon (hours) the scores come from


def weights_from_scores(score: pd.Series, vol: pd.Series, p: PortParams) -> pd.Series:
    """score/vol indexed by asset (hourly vol). Returns signed weights."""
    score = score.dropna()
    vol = vol.reindex(score.index)
    w = pd.Series(0.0, index=score.index)
    longs = score[score > p.thresh].nlargest(p.top_k)
    shorts = score[score < p.short_thresh].nsmallest(p.short_k) if p.short_k else score.iloc[0:0]
    for book, sign in ((longs, 1.0), (shorts, -1.0)):
        if book.empty:
            continue
        inv = 1.0 / vol[book.index].clip(lower=1e-4)
        w[book.index] += sign * inv / inv.sum()
    if (w != 0).sum() == 0:
        return w
    # scale to target vol assuming moderate correlation (0.5) between names
    av = vol * np.sqrt(HPY)
    var = ((w * av) ** 2).sum() + 0.5 * ((w * av).sum() ** 2 - ((w * av) ** 2).sum())
    pv = np.sqrt(max(var, 1e-12))
    w = w * (p.target_vol / pv)
    w = w.clip(-p.max_w, p.max_w)
    g = w.abs().sum()
    if g > p.gross_cap:
        w *= p.gross_cap / g
    return w


def spread_vector(spread_bps: float | pd.Series | Mapping | None, cols) -> np.ndarray:
    """Per-asset half-spread (bps) aligned to `cols`. None -> zeros; scalar -> broadcast;
    Series/dict -> reindexed, missing assets filled with the median of the given values."""
    if spread_bps is None:
        return np.zeros(len(cols))
    if isinstance(spread_bps, Mapping) and not isinstance(spread_bps, pd.Series):
        spread_bps = pd.Series(spread_bps, dtype=float)
    if isinstance(spread_bps, pd.Series):
        s = pd.to_numeric(spread_bps, errors="coerce")
        fill = float(s.median()) if s.notna().any() else 0.0
        return s.reindex(cols).fillna(fill).to_numpy(dtype=float)
    return np.full(len(cols), float(spread_bps))


def simulate(pred: pd.DataFrame, vol: pd.DataFrame, lc: pd.DataFrame, p: PortParams,
             rp: RiskParams = RiskParams(), fee: float = TAKER, mkt: pd.Series | None = None,
             weight_fn=None, maker_frac: float = 0.0,
             spread_bps: float | pd.Series | Mapping | None = None, maker_fee: float = MAKER):
    """pred/vol/lc: DataFrames indexed by hour, columns = assets (NaN = not tradeable).
    `fee` is the taker rate; see the module docstring for `maker_frac` / `spread_bps`.
    Returns equity Series and trades DataFrame (per rebalance: turnover, gross, n_long,
    n_short, fee_cost, spread_cost as fractions of equity).
    Numpy inner loop; pandas only at rebalances."""
    cols = pred.columns
    px = np.exp(lc[cols].values)
    with np.errstate(invalid="ignore", divide="ignore"):
        rets = np.nan_to_num(px[1:] / px[:-1] - 1.0, nan=0.0, posinf=0.0, neginf=0.0)
    rets = np.vstack([np.zeros((1, len(cols))), rets])
    P, V = pred.values, vol[cols].values
    M = None if mkt is None else mkt.values
    # fill model: blended fee rate, spread paid only on the taker fraction
    maker_frac = float(min(max(maker_frac, 0.0), 1.0))
    eff_fee = maker_frac * maker_fee + (1.0 - maker_frac) * fee
    half_spread = spread_vector(spread_bps, cols) * 1e-4 * (1.0 - maker_frac)
    w = np.zeros(len(cols))
    eq = 1.0
    rolling = RollingPeak(rp.window_h)
    curve = np.empty(len(pred))
    trades = []
    idx = pred.index
    for i in range(len(idx)):
        r = rets[i]
        port_r = float(w @ r)
        eq *= 1 + port_r
        if 1 + port_r > 0:
            w = w * (1 + r) / (1 + port_r)
        peak = rolling.update(eq)
        if i % p.rebal_h == 0:
            scale = exposure_scale(eq / peak - 1, rp)
            s = pd.Series(P[i], index=cols)
            if p.use_regime and M is not None and M[i] < 0:
                s = s.where(s < 0)  # longs off, shorts allowed
            wf = weight_fn or (lambda sc, vo, pp, ii: weights_from_scores(sc, vo, pp))
            tgt = wf(s, pd.Series(V[i], index=cols), p, i).reindex(cols).fillna(0.0).values * scale
            diff = tgt - w
            trade = np.where(np.abs(diff) > p.band, diff, 0.0)
            exit_mask = (tgt == 0) & (w != 0)
            trade[exit_mask] = -w[exit_mask]
            flip = np.sign(tgt) * np.sign(w) < 0
            trade[flip] = diff[flip]
            to = float(np.abs(trade).sum())
            if to > 0:
                short_to = float(np.abs(trade[(w + trade < 0) | (w < 0)]).sum())
                fee_cost = to * eff_fee + short_to * max(SHORT_FEE - eff_fee, 0.0)
                spread_cost = float(np.abs(trade) @ half_spread)
                eq *= 1 - fee_cost - spread_cost
                w = w + trade
                trades.append({"t": idx[i], "turnover": to, "gross": float(np.abs(w).sum()),
                               "n_long": int((w > 0).sum()), "n_short": int((w < 0).sum()),
                               "fee_cost": fee_cost, "spread_cost": spread_cost})
        curve[i] = eq
    return pd.Series(curve, index=idx), pd.DataFrame(trades)


def report(name: str, equity: pd.Series, btc: pd.Series | None = None, days: int = 14) -> dict:
    """Full-period metrics + non-overlapping 14d windows (win14_*) + rolling daily-step
    14d window distribution from bot.ml.windows.summarize (w14_*; the primary objective)."""
    m = metrics(equity)
    wr = window_report(equity, days)
    out = {"name": name, **{k: round(float(v), 3) for k, v in m.items()}}
    if wr.empty:
        out.update({"win14_med_ret": float("nan"), "win14_med_score": float("nan"), "win14_pos": float("nan")})
    else:
        out.update({"win14_med_ret": round(float(wr["return"].median()), 4),
                    "win14_med_score": round(float(wr["score"].median()), 2),
                    "win14_pos": round(float((wr["return"] > 0).mean()), 2)})
    for k, v in summarize(equity, days).items():
        if k == "n_windows":
            out[f"w14_{k}"] = int(v)
        else:
            out[f"w14_{k}"] = round(float(v), 2 if k.startswith("score") else 4)
    return out
