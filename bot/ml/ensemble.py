"""Rank-average the predictions of several trained models into a pseudo-tag (default
'ens') so evaluate/stability/diagnose/coretilt can treat it like a single model.
Member predictions that are not cached yet are generated from models/<tag>.pt.

    python -m bot.ml.ensemble cnn64 cnn32                       # -> pred_ens_*
    python -m bot.ml.ensemble cnn64s0 cnn64s1 cnn64s2 --out ens24
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from bot.ml.evaluate import RESULTS, predictions
from bot.ml.features import HORIZONS

SPLITS = ("val", "test", "holdout")


def member_predictions(tags: list[str]) -> None:
    """Make sure results/pred_<tag>_<split>_<h>.parquet exists for every member."""
    missing = [t for t in tags for s in SPLITS for h in HORIZONS if not (RESULTS / f"pred_{t}_{s}_{h}.parquet").exists()]
    if not missing:
        return
    from bot.ml.dataset import get_panel
    feat, vol, lc, valid, times, coins = get_panel()
    for t in sorted(set(missing)):
        for s in SPLITS:
            predictions(t, s, feat, valid, times, coins)
            print(f"predicted {t} {s}", flush=True)


def build(tags: list[str], out: str = "ens") -> None:
    member_predictions(tags)
    for split in SPLITS:
        for h in HORIZONS:
            frames = [pd.read_parquet(RESULTS / f"pred_{t}_{split}_{h}.parquet") for t in tags]
            ens = sum(f.rank(axis=1, pct=True) for f in frames) / len(frames) - 0.5  # centred in [-0.5, 0.5]
            # express in roughly the same units as a single model (vol-normalised return) for thresholds
            ens = ens * 2 * pd.concat(frames).stack().std()
            ens.to_parquet(RESULTS / f"pred_{out}_{split}_{h}.parquet")
    (RESULTS / f"{out}_members.json").write_text(json.dumps(tags))
    print(f"ensemble '{out}' written for", tags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*", default=["cnn64", "cnn32"])
    ap.add_argument("--out", default="ens", help="pseudo-tag name of the ensemble")
    args = ap.parse_args()
    build(args.tags, args.out)


if __name__ == "__main__":
    main()
