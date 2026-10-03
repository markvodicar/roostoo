"""Candidate proposal backtest: regime-switched blend, no drawdown brake.

  bull  (BTC > 200d MA and breadth > 0.5): 50% majors (BTC/ETH/SOL/BNB/XRP equal weight)
                                           + 25% liquid-20 trend (vol target 1.0)
                                           + 25% top-5 30d momentum among liquid-20
  mixed (otherwise)                       : 100% liquid-20 trend (vol target 0.5)
  bear  (BTC < 200d MA and breadth < 0.3) : gated BTC/ETH core with BTC short (vol 0.3)

    python3 -m bot.research.proposal
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot.research.regime_study import (START, describe, load_daily, regime_flags, simulate,
                                       strategies, window_stats)


def full_metrics(eq):
    r = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365
    tot = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (1 + tot) ** (1 / yrs) - 1
    sh = r.mean() / r.std() * np.sqrt(365)
    so = r.mean() / np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(365)
    mdd = (eq / eq.cummax() - 1).min()
    return {"total": tot, "CAGR": cagr, "Sharpe": sh, "Sortino": so, "Calmar": cagr / abs(mdd),
            "MaxDD": mdd, "vol": r.std() * np.sqrt(365)}


def build():
    C, V, seen = load_daily()
    S, R, aux = strategies(C, V, seen)
    f = regime_flags(C, aux)
    bull = (f.btc_gt_200d & (f.breadth50 > 0.5)).astype(float)
    bear = ((~f.btc_gt_200d) & (f.breadth50 < 0.3)).astype(float)
    mixed = 1 - bull - bear
    M, T, T1, X, GS = (S[k][0] for k in ("majors5_ew_hold", "trend_liquid20", "trend_liquid20_v1",
                                         "xsmom_top5", "core_gated_short"))
    W = (M * 0.5 + T1 * 0.25 + X * 0.25).mul(bull, axis=0) + T.mul(mixed, axis=0) + GS.mul(bear, axis=0)
    return C, R, f, bull, {"PROPOSAL regime blend": W, "launch core_gated_short": GS,
                           "btc_hold": S["btc_hold"][0], "majors5_ew_hold": M}


def main():
    C, R, f, bull, books = build()
    idx = C.index[(C.index >= START) & (C.index <= C.index[-1] - pd.Timedelta(days=14))]
    sets = {"all": idx, "bull_regime": idx[bull.reindex(idx).values > 0],
            "now_like": idx[f.now_like.reindex(idx).fillna(False).values],
            "last_365d": idx[idx >= C.index[-1] - pd.Timedelta(days=365)]}
    pd.set_option("display.width", 250)
    rows, full = [], []
    for name, W in books.items():
        eq = simulate(W, R, 1)
        eq = eq[eq.index >= START]
        to = (W.diff().abs().sum(1))[W.index >= START].sum() / ((eq.index[-1] - eq.index[0]).days / 365)
        for yr0, yr1 in (("2019", "2027"), ("2019", "2021"), ("2021", "2022"), ("2022", "2023"),
                         ("2023", "2025"), ("2025", "2027")):
            seg = eq[(eq.index >= yr0) & (eq.index < yr1)]
            full.append({"strategy": name, "period": f"{yr0}-{int(yr1) - 1}", **full_metrics(seg / seg.iloc[0])})
        for sn, st in sets.items():
            rows.append({"strategy": name, "set": sn, **describe(window_stats(eq, st)), "turnover_yr": to})
    print(pd.DataFrame(full).set_index(["period", "strategy"]).round(3).to_string())
    df = pd.DataFrame(rows)
    for sn in sets:
        print(f"\n--- 14d windows: {sn} ---")
        print(df[df.set == sn].drop(columns="set").set_index("strategy").round(3).to_string())
    pd.DataFrame(full).to_csv("results/proposal_full.csv", index=False)
    df.to_csv("results/proposal_windows.csv", index=False)
    W = books["PROPOSAL regime blend"].iloc[-1]
    print("\nProposal weights today:", W[W.abs() > 0.005].round(3).sort_values(ascending=False).to_dict(),
          "gross", round(W.abs().sum(), 2))


if __name__ == "__main__":
    main()
