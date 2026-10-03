"""Diagnostic (NOT selection): how much of the P&L is eaten by fees, and how do
low-turnover long-only configs behave on val and test. Everything printed here is
test-informed, so nothing chosen from it may be claimed as out-of-sample on test;
the 2026 holdout remains the only clean check after this.

    python -m bot.ml.diagnose --tag cnn64
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from bot.backtest import metrics
from bot.fees import TAKER
from bot.ml.dataset import get_panel
from bot.ml.evaluate import frames_for, predictions
from bot.ml.features import HORIZONS
from bot.ml.portfolio import MEDIAN_SPREAD_BPS, PortParams, simulate

# execution assumptions: no fees / all-taker (rule 8) / limit-first with half the notional as maker + spread
FILLS = {"nofee": dict(fee=0.0), "taker": dict(fee=TAKER),
         "maker50+spread": dict(fee=TAKER, maker_frac=0.5, spread_bps=MEDIAN_SPREAD_BPS)}

CONFIGS = {
    "val_best(72h,L3/S2,12h)": PortParams(top_k=3, short_k=2, thresh=0.1, short_thresh=-0.1, target_vol=0.25, rebal_h=12, head=72),
    "long_only 72h k5 12h": PortParams(top_k=5, head=72, rebal_h=12, target_vol=0.25),
    "long_only 336h k5 24h b.05": PortParams(top_k=5, head=336, rebal_h=24, band=0.05, target_vol=0.25),
    "long_only 336h k8 24h b.05": PortParams(top_k=8, head=336, rebal_h=24, band=0.05, target_vol=0.25),
    "long_only 336h k5 48h b.10": PortParams(top_k=5, head=336, rebal_h=48, band=0.10, target_vol=0.25),
    "long_only 336h k5 48h b.10 regime": PortParams(top_k=5, head=336, rebal_h=48, band=0.10, target_vol=0.25, use_regime=True),
    "long_only 336h k5 48h b.10 tv.4": PortParams(top_k=5, head=336, rebal_h=48, band=0.10, target_vol=0.40),
    "long_only 24h k5 24h b.05": PortParams(top_k=5, head=24, rebal_h=24, band=0.05, target_vol=0.25),
    "L/S 336h k5/2 48h b.10": PortParams(top_k=5, short_k=2, head=336, rebal_h=48, band=0.10, target_vol=0.25),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cnn64")
    args = ap.parse_args()
    feat, vol, lc, valid, times, coins = get_panel()
    rows = []
    for split in ("val", "test"):
        preds = predictions(args.tag, split, feat, valid, times, coins)
        volf, lcf, mkt = frames_for(split, preds[HORIZONS[0]], vol, lc, feat, times, coins)
        years = (lcf.index[-1] - lcf.index[0]).days / 365
        for name, p in CONFIGS.items():
            for fill, kw in FILLS.items():
                eq, tr = simulate(preds[p.head], volf, lcf, p, mkt=mkt, **kw)
                m = metrics(eq)
                rows.append({"split": split, "config": name, "fill": fill, "ret": round(m["return"], 3),
                             "score": round(m["score"], 2), "maxdd": round(m["max_dd"], 3),
                             "turnover/yr": round(tr["turnover"].sum() / years, 1) if len(tr) else 0,
                             "avg_gross": round(tr["gross"].mean(), 2) if len(tr) else 0})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    piv = df.pivot_table(index=["config"], columns=["split", "fill"], values=["ret", "score"])
    print(piv.round(2).to_string())
    print("\nturnover/yr and avg gross exposure (taker run):")
    print(df[df.fill == "taker"].pivot_table(index="config", columns="split", values=["turnover/yr", "avg_gross"]).round(2).to_string())


if __name__ == "__main__":
    main()
