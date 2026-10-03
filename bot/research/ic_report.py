"""Information coefficient per forecast head per split for one or more prediction tags.

Hourly cross-sectional Spearman rank correlation between the predicted and realised
vol-normalised forward return (same definition as bot.ml.evaluate), with a t-statistic
that only counts non-overlapping horizons.

    python -m bot.research.ic_report --tags ens ens24 cnn64s0
Writes results/research_ic.csv.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from bot.ml.dataset import get_panel
from bot.ml.evaluate import RESULTS, frames_for, predictions
from bot.ml.features import HORIZONS

SPLITS = ("val", "test", "holdout")


def ic_table(tags: list[str], splits=SPLITS) -> pd.DataFrame:
    feat, vol, lc, valid, times, coins = get_panel()
    coins = [str(c) for c in coins]
    rows = []
    for tag in tags:
        for split in splits:
            preds = predictions(tag, split, feat, valid, times, coins)
            volf, lcf, _ = frames_for(split, preds[HORIZONS[0]], vol, lc, feat, times, coins)
            for h in HORIZONS:
                fwd = (lcf.shift(-h) - lcf) / (volf * np.sqrt(h))
                ic = pd.concat([preds[h].stack(), fwd.stack()], axis=1).dropna()
                ic.columns = ["p", "y"]
                ic_t = ic.groupby(level=0).apply(lambda g: g.p.corr(g.y, method="spearman") if len(g) >= 5 else np.nan)
                n_eff = ic_t.notna().sum() / h
                rows.append({"tag": tag, "split": split, "head": h, "ic": float(ic_t.mean()),
                             "ic_t": float(ic_t.mean() / ic_t.std() * np.sqrt(n_eff)),
                             "ic_pos_frac": float((ic_t > 0).mean()),
                             "coverage": float(preds[h].notna().mean().mean()), "n_hours": int(ic_t.notna().sum())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["ens", "ens24"])
    ap.add_argument("--out", default="research_ic")
    args = ap.parse_args()
    df = ic_table(args.tags)
    pd.set_option("display.width", 200)
    piv = df.pivot_table(index=["tag", "head"], columns="split", values=["ic", "ic_t"]).reindex(
        columns=pd.MultiIndex.from_product([["ic", "ic_t"], list(SPLITS)]))
    print(piv.round(4).to_string())
    df.to_csv(RESULTS / f"{args.out}.csv", index=False)


if __name__ == "__main__":
    main()
