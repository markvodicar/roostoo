"""Robust selection of the pure-core configuration from the compare_tilts grid.

A lexicographic "best ret_med" pick is degenerate for a gated core: a core that is
flat in more than half of the 14-day windows has ret_med exactly 0 and wins on a
technicality while giving up the upside the competition's top-20 return gate needs.
Instead we rank every config by the AVERAGE PERCENTILE RANK across val and test of
four window statistics (ret_med, p_pos, p_gt3, ret_p10), so a config must be decent
on a bear year AND a bull period, on upside AND tail. Holdout is reported, not used.

    python -m bot.research.select_core --out ens [--family core] [--top 15]
"""
from __future__ import annotations

import argparse

import pandas as pd

from bot.ml.evaluate import RESULTS

KEYS = ["ret_med", "p_pos", "p_gt3", "ret_p10"]
SHOW = ["ret_med", "ret_p10", "p_pos", "p_gt3", "maxdd_med", "score_med_if_pos", "return", "turnover/yr"]


def robust_rank(df: pd.DataFrame, families: list[str], keys=KEYS, splits=("val", "test")) -> pd.DataFrame:
    d = df[df.family.isin(families)]
    parts = []
    for split in splits:
        x = d[d.split == split].set_index("name")[keys]
        parts.append(x.rank(pct=True).mean(axis=1).rename(f"rank_{split}"))
    r = pd.concat(parts, axis=1)
    r["robust"] = r.mean(axis=1)
    r["min_split"] = r[[f"rank_{s}" for s in splits]].min(axis=1)
    return r.sort_values(["robust", "min_split"], ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ens")
    ap.add_argument("--family", nargs="+", default=["core"])
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--keys", nargs="+", default=KEYS, help="window statistics to average the percentile rank over")
    ap.add_argument("--tag", default="", help="suffix for the output csv")
    args = ap.parse_args()
    df = pd.read_csv(RESULTS / f"research_tilts_{args.out}.csv")
    r = robust_rank(df, args.family, args.keys)
    pd.set_option("display.width", 250)
    print(f"=== robust ranking of {args.family} configs (mean pct-rank of {args.keys} over val+test) ===")
    print(r.head(args.top).round(3).to_string())
    best = r.index[0]
    print(f"\nselected: {best}")
    for split in ("val", "test", "holdout"):
        row = df[(df.name == best) & (df.split == split)].iloc[0]
        print(f"  {split:8s} " + "  ".join(f"{k}={row[k]:+.4f}" if k in ("ret_med", "ret_p10") else f"{k}={row[k]:.3f}" for k in SHOW))
    out = RESULTS / f"research_select_{args.out}{('_' + args.tag) if args.tag else ''}.csv"
    r.to_csv(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
