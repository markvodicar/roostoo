"""Risk-balanced trend portfolio (long-only, no leverage) - the Sharpe-first strategy.

Single source of truth for the weights: the backtest (bot/research/sharpe_opt.py) and
the live bot (RiskParitySignal) both call `target_weights`, so what is backtested is
what is traded.

At each daily close t (00:00 UTC), using data up to and including t:
  trend_i = mean over lookbacks L of 1[close_i > SMA_L(close_i)]          in [0, 1]
  vol_i   = EWMA daily vol, halflife h, annualised sqrt(365)
  raw_i   = budget_i * trend_i / vol_i                                   risk parity x trend
  k       = min(target_vol / ex-ante portfolio vol, 1 / sum(raw))        vol target, no leverage
  w_i     = k * raw_i
Ex-ante portfolio vol uses an EWMA covariance with the same halflife.
Rationale (results/sharpe_summary.md): volatility clusters, so scaling exposure inversely
to forecast vol raises Sharpe; the 200-day trend filter removes most of the deep bear
legs; gold (PAXG) has near-zero correlation to crypto and holds up when crypto falls.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ANN = 365


@dataclass
class RPParams:
    sleeves: list[str] = field(default_factory=lambda: ["BTC", "ETH", "PAXG"])
    budgets: list[float] = field(default_factory=lambda: [2.0, 2.0, 1.0])
    lookbacks: list[int] = field(default_factory=lambda: [200])     # days
    halflife: int = 30                                               # days, vol and covariance
    target_vol: float = 0.20                                         # annualised
    band: float = 0.02                                               # min |w change| to trade a sleeve
    min_obs: int = 60                                                # days of history before trading
    trend_floor: float = 0.0           # weight multiplier when an asset's trend is off (0 = sell it all)
    force_daily_trade: bool = False    # if nothing filled for a day, ignore the band at the next rebalance
    min_trade_frac: float = 0.00025    # smallest trade, as a fraction of equity ($25 at $100k)
    rebal_h: int = 24                  # hours between rebalances to the (daily) targets
    max_gross: float = 1.0             # cap on total exposure (< 1 keeps USD for fees and small buys)

    def validate(self) -> "RPParams":
        if len(self.sleeves) != len(self.budgets) or not self.sleeves:
            raise ValueError("sleeves and budgets must be non-empty and the same length")
        if min(self.budgets) < 0 or self.target_vol <= 0 or self.halflife < 1:
            raise ValueError("budgets >= 0, target_vol > 0, halflife >= 1 required")
        if not self.lookbacks or min(self.lookbacks) < 2:
            raise ValueError("lookbacks must be >= 2 days")
        if self.rebal_h < 1 or 24 % self.rebal_h:
            raise ValueError("rebal_h must divide 24")
        if not 0.0 < self.max_gross <= 1.0:
            raise ValueError("max_gross must be in (0, 1]")
        if not 0.0 <= self.trend_floor <= 1.0:
            raise ValueError("trend_floor must be in [0, 1]")
        return self


def trend_signal(P: pd.DataFrame, lookbacks) -> pd.DataFrame:
    s = sum((P > P.rolling(L, min_periods=L).mean()).astype(float) for L in lookbacks) / len(lookbacks)
    return s.where(P.notna())


def _vol_targeted(P: pd.DataFrame, p: RPParams, sig: pd.DataFrame) -> np.ndarray:
    """Risk parity x signal, scaled to the volatility target, gross capped at 1."""
    R = P.pct_change(fill_method=None)
    vol = np.sqrt((R ** 2).ewm(halflife=p.halflife, min_periods=20).mean()) * np.sqrt(ANN)
    raw = (sig * pd.Series(p.budgets, index=p.sleeves) / vol).fillna(0.0).values
    rv = R.fillna(0.0).values
    lam = 0.5 ** (1 / p.halflife)
    S = np.zeros((len(p.sleeves), len(p.sleeves)))
    W = np.zeros_like(raw)
    for i in range(len(P)):
        S = lam * S + (1 - lam) * np.outer(rv[i], rv[i])
        w = raw[i]
        if i < p.min_obs or w.sum() <= 0:
            continue
        pv = float(np.sqrt(max(w @ S @ w, 1e-16)) * np.sqrt(ANN))
        W[i] = w * min(p.target_vol / pv, p.max_gross / w.sum())
    return W


def target_weights(P: pd.DataFrame, p: RPParams, trend: bool = True,
                   signal: pd.DataFrame | None = None) -> pd.DataFrame:
    """P: daily closes (rows = days, columns = p.sleeves). Returns weights per day (rows),
    decided at that day's close. Rows before `min_obs` days of history are zero.
    `signal` overrides the trend signal (research placebos only; the live bot never sets it).

    trend_floor f > 0: an asset whose trend is off keeps f x the weight it would have if
    every trend were on (applied after the volatility scaling, so a market where every
    trend is off holds only a small position rather than a re-levered one)."""
    P = P[p.sleeves]
    if signal is not None:
        sig = signal[p.sleeves].reindex(P.index)
    else:
        sig = trend_signal(P, p.lookbacks) if trend else P.notna().astype(float)
    W = _vol_targeted(P, p, sig)
    if p.trend_floor > 0:
        on = P.notna().astype(float)
        W_all = _vol_targeted(P, p, on)
        W = W + p.trend_floor * W_all * (1.0 - sig.fillna(0.0).values)
        g = W.sum(1, keepdims=True)
        W = np.where(g > p.max_gross, W * p.max_gross / np.maximum(g, 1e-12), W)
    return pd.DataFrame(W, index=P.index, columns=p.sleeves)


def simulate(P: pd.DataFrame, W: pd.DataFrame, band: float, fee: float, force_daily: bool = False,
             min_trade: float = 0.0, trade_log: list | None = None) -> pd.Series:
    """Daily returns of trading W with a per-sleeve no-trade band; weights set at close t
    earn t -> t+1. Exits (target 0) always trade. With `force_daily`, a day after a day with
    no trade uses band 0 (every difference above `min_trade` is traded), mirroring the live
    activity guarantee. `trade_log`, if given, receives the number of legs traded per day."""
    R = P[W.columns].pct_change(fill_method=None).fillna(0.0).values
    Wv = W.values
    cur = np.zeros(W.shape[1])
    eq, out = 1.0, np.empty(len(W))
    traded_prev = True
    for i in range(len(W)):
        r = R[i]
        pr = float(cur @ r)
        eq *= 1 + pr
        if 1 + pr > 0:
            cur = cur * (1 + r) / (1 + pr)
        tgt = Wv[i]
        diff = tgt - cur
        b = min_trade if (force_daily and not traded_prev) else band
        trade = np.where(np.abs(diff) > max(b, min_trade), diff, 0.0)
        trade = np.where((tgt == 0) & (cur != 0), -cur, trade)
        to = np.abs(trade).sum()
        if to > 0:
            eq *= 1 - to * fee
            cur = cur + trade
        traded_prev = to > 0
        if trade_log is not None:
            trade_log.append(int((trade != 0).sum()))
        out[i] = eq
    eqs = pd.Series(out, index=W.index)
    return eqs.pct_change().fillna(0.0)
