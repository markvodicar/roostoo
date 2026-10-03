"""Regime-switching combinations of the strategy families in regime_study.

Regimes (decided daily at close, no look-ahead):
  bull : BTC > 200d MA and breadth (share of liquid-20 coins above EMA50) > 0.5
  bear : BTC < 200d MA and breadth < 0.3
  mixed: otherwise
Books are blended per regime. Every parameter is a round number fixed in advance.

    python3 -m bot.research.regime_combo
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot.research.regime_study import (START, describe, load_daily, regime_flags, simulate,
                                       strategies, window_stats)


def blend(parts, mask=None):
    w = sum(W * f for W, f in parts)
    return w if mask is None else w.mul(mask, axis=0)


def dd_brake(eq_fn, W, R, rebal, soft=0.05, hard=0.10, floor=0.3, window=14):
    """Two-pass drawdown brake: simulate, measure drawdown from rolling 14d peak at each
    close, scale next day's weights (uses only past equity)."""
    eq = eq_fn(W, R, rebal)
    dd = eq / eq.rolling(window, min_periods=1).max() - 1
    s = (1 - ((-dd - soft) / (hard - soft)).clip(0, 1) * (1 - floor))
    return W.mul(s, axis=0)


def main():
    C, V, seen = load_daily()
    S, R, aux = strategies(C, V, seen)
    f = regime_flags(C, aux)
    bull = (f.btc_gt_200d & (f.breadth50 > 0.5)).astype(float)
    bear = ((~f.btc_gt_200d) & (f.breadth50 < 0.3)).astype(float)
    mixed = 1 - bull - bear
    T, X, G, GS = (S[k][0] for k in ("trend_liquid20", "xsmom_top5", "core_gated", "core_gated_short"))
    T1 = S["trend_liquid20_v1"][0]
    combos = {
        "trend_liquid20": T,
        "trend+xsmom_50_50": blend([(T, 0.5), (X, 0.5)]),
        "trend_v1+xsmom_50_50": blend([(T1, 0.5), (X, 0.5)]),
        "REGIME: bull trend+xsmom / mixed core / bear core+short":
            blend([(T, 0.5), (X, 0.5)], bull) + G.mul(mixed, axis=0) + GS.mul(bear, axis=0),
        "REGIME v1: bull trend_v1+xsmom / mixed trend / bear core+short":
            blend([(T1, 0.5), (X, 0.5)], bull) + T.mul(mixed, axis=0) + GS.mul(bear, axis=0),
        "launch: core_gated_short": GS,
        "btc_hold": S["btc_hold"][0],
        "majors5_ew_hold": S["majors5_ew_hold"][0],
    }
    idx = C.index[(C.index >= START) & (C.index <= C.index[-1] - pd.Timedelta(days=14))]
    now_like = f.now_like.reindex(idx).fillna(False).values
    sets = {"all_2019_26": idx, "now_like": idx[now_like],
            "bull_regime": idx[bull.reindex(idx).values > 0],
            "last_365d": idx[idx >= C.index[-1] - pd.Timedelta(days=365)],
            "2019-20": idx[idx < "2021-01-01"], "2021": idx[(idx >= "2021-01-01") & (idx < "2022-01-01")],
            "2022": idx[(idx >= "2022-01-01") & (idx < "2023-01-01")],
            "2023-24": idx[(idx >= "2023-01-01") & (idx < "2025-01-01")],
            "2025-26": idx[idx >= "2025-01-01"]}
    rows, curves = [], {}
    for name, W in combos.items():
        for brake in (False, True):
            Wb = dd_brake(simulate, W, R, 1) if brake else W
            eq = simulate(Wb, R, 1)
            lbl = name + (" +brake" if brake else "")
            curves[lbl] = eq
            for sn, st in sets.items():
                rows.append({"strategy": lbl, "set": sn, **describe(window_stats(eq, st))})
    df = pd.DataFrame(rows)
    df.to_csv("results/regime_combo.csv", index=False)
    pd.DataFrame(curves).to_parquet("results/regime_combo_eq.parquet")
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
    for sn in ("all_2019_26", "now_like", "bull_regime", "last_365d"):
        print(f"\n=== {sn} ===")
        print(df[df.set == sn].drop(columns="set").set_index("strategy").round(3).to_string())
    print("\n=== median 14d return by period ===")
    print(df[df.set.isin(["2019-20", "2021", "2022", "2023-24", "2025-26"])]
          .pivot(index="strategy", columns="set", values="ret_med").round(3).to_string())
    print("\n=== p(14d return > 0) by period ===")
    print(df[df.set.isin(["2019-20", "2021", "2022", "2023-24", "2025-26"])]
          .pivot(index="strategy", columns="set", values="p_pos").round(2).to_string())
    print("\nregime today:", "bull" if bull.iloc[-1] else ("bear" if bear.iloc[-1] else "mixed"),
          "| share of days 2019-26: bull %.2f mixed %.2f bear %.2f" % (bull[idx].mean(), mixed[idx].mean(), bear[idx].mean()))


if __name__ == "__main__":
    main()
