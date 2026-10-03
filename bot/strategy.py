"""Regime-filtered, volatility-scaled cross-sectional momentum (long-only spot).

The same `target_weights` function drives both the backtest and the live bot, so
what we backtest is exactly what we trade.

Idea
----
* Crypto has strong short/medium-horizon time-series and cross-sectional momentum,
  but it crashes together. So: hold the strongest few coins only when the market is
  trending up, and sit in USD otherwise (USD is our zero-vol "defensive asset").
* Score = blend of risk-adjusted returns over several lookbacks (vol-normalised so a
  meme coin's +20% doesn't automatically beat BTC's +5%).
* Weights are inverse-vol, then scaled so the whole book targets a fixed annualised
  volatility. Lower vol => better Sharpe/Sortino/Calmar, which is 60%+ of the score.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 24 * 365


@dataclass
class Params:
    lookbacks: tuple[int, ...] = (24, 72, 168)  # hours
    vol_window: int = 72                         # hours, for realised vol
    trend_ema: int = 96                          # per-asset trend filter
    regime_ema: int = 168                        # BTC regime filter
    breadth_min: float = 0.40                    # min share of universe above its EMA
    top_k: int = 5
    max_weight: float = 0.30                     # per asset, fraction of equity
    target_vol: float = 0.35                     # annualised portfolio vol target
    gross_cap: float = 1.0                       # no leverage
    min_history: int = 24 * 10                   # bars before an asset is eligible
    exclude: tuple[str, ...] = ("PAXG/USD",)     # gold token: no momentum edge, low vol
    regime_asset: str = "BTC/USD"


def target_weights(closes: pd.DataFrame, p: Params = Params()) -> pd.Series:
    """closes: hourly close panel (index time, columns pairs), last row = now.
    Returns target weights (fraction of equity) per pair; remainder is USD."""
    closes = closes.drop(columns=[c for c in p.exclude if c in closes.columns])
    need = max(max(p.lookbacks), p.vol_window, p.trend_ema, p.regime_ema) + 1
    window = closes.iloc[-max(need, p.min_history) - 5:]
    zero = pd.Series(0.0, index=closes.columns)
    if len(window) < need:
        return zero

    eligible = window.notna().sum() >= min(p.min_history, len(window))
    eligible &= window.iloc[-1].notna()
    px = window.loc[:, eligible].ffill()
    if px.shape[1] == 0:
        return zero

    rets = np.log(px).diff()
    vol = rets.iloc[-p.vol_window:].std() * np.sqrt(HOURS_PER_YEAR)
    vol = vol.replace(0, np.nan)

    # --- regime: BTC trend + market breadth ---
    ema = px.ewm(span=p.trend_ema, adjust=False).mean()
    above = px.iloc[-1] > ema.iloc[-1]
    breadth = above.mean()
    risk_on = breadth >= p.breadth_min
    if p.regime_asset in px.columns:
        btc = px[p.regime_asset]
        risk_on &= btc.iloc[-1] > btc.ewm(span=p.regime_ema, adjust=False).mean().iloc[-1]
    if not risk_on:
        return zero

    # --- score: vol-normalised multi-horizon momentum ---
    score = pd.Series(0.0, index=px.columns)
    for lb in p.lookbacks:
        r = np.log(px.iloc[-1] / px.iloc[-1 - lb])
        score += (r / (vol * np.sqrt(lb / HOURS_PER_YEAR))).fillna(-np.inf) / len(p.lookbacks)

    cand = score[(score > 0) & above & vol.notna()].sort_values(ascending=False).head(p.top_k)
    if cand.empty:
        return zero

    # --- sizing: inverse vol, then portfolio vol target ---
    inv = 1.0 / vol[cand.index]
    w = inv / inv.sum()
    cov = rets[cand.index].iloc[-p.vol_window:].cov() * HOURS_PER_YEAR
    port_vol = float(np.sqrt(w.values @ cov.values @ w.values)) if len(w) > 1 else float(vol[cand.index[0]])
    scale = p.target_vol / port_vol if port_vol > 0 else 0.0
    w = (w * scale).clip(upper=p.max_weight)
    if w.sum() > p.gross_cap:
        w *= p.gross_cap / w.sum()
    return w.reindex(closes.columns).fillna(0.0)
