"""Hourly event-driven backtest with Roostoo's fee model and the competition score.

Fees come from bot/fees.py (competition rule 8): taker 0.10% on every traded notional
(we assume MARKET orders, i.e. the worst case; maker is 0.05%). Trades happen at the
next bar's close after the signal to avoid look-ahead, and only when a position drifts
more than `rebalance_band` of equity from target, which keeps turnover (and fees) down.
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

from bot.data import load_panel
from bot.fees import MAKER, TAKER
from bot.risk import RiskParams, RollingPeak, exposure_scale
from bot.strategy import Params, target_weights

TAKER_FEE = TAKER  # backward-compatible aliases; single source of truth is bot/fees.py
MAKER_FEE = MAKER


def metrics(equity: pd.Series) -> dict:
    """Competition-style metrics on daily equity (annualised, crypto 365d year)."""
    daily = equity.resample("1D").last().dropna()
    r = daily.pct_change(fill_method=None).dropna()
    ann = 365
    total = equity.iloc[-1] / equity.iloc[0] - 1
    n_days = max((equity.index[-1] - equity.index[0]).total_seconds() / 86400, 1e-9)
    ann_ret = (1 + total) ** (ann / n_days) - 1
    sd = r.std()
    dd_sd = np.sqrt((np.minimum(r, 0) ** 2).mean())
    max_dd = float((equity / equity.cummax() - 1).min())
    sharpe = r.mean() / sd * np.sqrt(ann) if sd > 0 else 0.0
    sortino = r.mean() / dd_sd * np.sqrt(ann) if dd_sd > 0 else 0.0
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0
    return {"return": total, "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
            "max_dd": max_dd, "score": 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar}


def run(closes: pd.DataFrame, p: Params = Params(), rp: RiskParams = RiskParams(),
        fee: float = TAKER_FEE, rebalance_band: float = 0.03, start_equity: float = 1.0,
        warmup: int | None = None) -> tuple[pd.Series, pd.DataFrame]:
    warmup = warmup or max(max(p.lookbacks), p.trend_ema, p.regime_ema, p.min_history) + 5
    px = closes.ffill()
    rets = px.pct_change(fill_method=None).fillna(0.0)
    w = pd.Series(0.0, index=px.columns)
    eq = start_equity
    rolling = RollingPeak(rp.window_h)
    eq_curve, trades = [], []
    idx = px.index
    for i in range(warmup, len(idx)):
        # 1) mark to market over bar i (positions decided at close of i-1)
        port_r = float((w * rets.iloc[i]).sum())
        eq *= 1 + port_r
        # weights drift with prices
        if port_r != -1:
            w = w * (1 + rets.iloc[i]) / (1 + port_r)
        peak = rolling.update(eq)
        # 2) decide new targets at close of bar i
        tgt = target_weights(closes.iloc[: i + 1], p) * exposure_scale(eq / peak - 1, rp)
        diff = tgt - w
        trade = diff.where(diff.abs() > rebalance_band, 0.0)
        # always fully exit names dropped from target
        trade[(tgt == 0) & (w > 0)] = -w[(tgt == 0) & (w > 0)]
        turnover = float(trade.abs().sum())
        if turnover > 0:
            eq *= 1 - turnover * fee
            w = w + trade
            trades.append({"t": idx[i], "turnover": turnover, "gross": float(w.sum())})
        eq_curve.append((idx[i], eq))
    equity = pd.Series(dict(eq_curve))
    return equity, pd.DataFrame(trades)


def window_report(equity: pd.Series, days: int = 14) -> pd.DataFrame:
    rows = []
    start = equity.index[0]
    while start + pd.Timedelta(days=days) <= equity.index[-1]:
        seg = equity[start: start + pd.Timedelta(days=days)]
        rows.append({"start": start.date(), **metrics(seg)})
        start += pd.Timedelta(days=days)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--fee", type=float, default=TAKER_FEE)
    args = ap.parse_args()

    import requests
    info = requests.get("https://mock-api.roostoo.com/v3/exchangeInfo", timeout=10).json()
    pairs = sorted(k for k, v in info["TradePairs"].items() if v.get("AssetType") == "crypto")
    closes = load_panel(pairs)
    closes = closes[closes.index >= closes.index[-1] - pd.Timedelta(days=args.days)]

    eq, trades = run(closes, fee=args.fee)
    m = metrics(eq)
    btc = closes["BTC/USD"].loc[eq.index]
    print("Strategy :", {k: round(v, 3) for k, v in m.items()})
    print("BTC hold :", {k: round(v, 3) for k, v in metrics(btc / btc.iloc[0]).items()})
    print(f"Trades: {len(trades)}, avg hourly turnover {trades.turnover.mean():.3f}, "
          f"time invested {(trades.set_index('t').gross.reindex(eq.index).ffill().fillna(0) > 0).mean():.0%}")
    wr = window_report(eq)
    print("\n14-day windows:\n", wr.round(3).to_string(index=False))
    print("\nWindow medians:", wr.drop(columns="start").median().round(3).to_dict())


if __name__ == "__main__":
    main()
