"""Markdown tables for results/research_summary.md from the compare_tilts outputs.

    python -m bot.research.summary_tables --out ens        # reads results/research_tilts_ens.csv etc.
Prints markdown to stdout (pasted into results/research_summary.md).
"""
from __future__ import annotations

import argparse

import pandas as pd

from bot.ml.evaluate import RESULTS
from bot.research.compare_tilts import FAMILY_LABEL, RANKINGS, SPLITS, pick, row_of

COLS = ["ret_med", "ret_p10", "p_pos", "p_gt3", "maxdd_med", "score_med_if_pos", "return", "turnover/yr"]
HDR = {"ret_med": "w14 ret_med", "ret_p10": "w14 ret_p10", "p_pos": "p_pos", "p_gt3": "p>3%", "maxdd_med": "w14 maxdd_med",
       "score_med_if_pos": "score_med_if_pos", "return": "full return", "turnover/yr": "turnover/yr"}


def md(df: pd.DataFrame, cols: list[str], index: bool = False) -> str:
    d = df[cols].copy()
    for c in cols:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: f"{v:+.4f}" if c in ("ret_med", "ret_p10") else (f"{v:.0f}" if c in ("turnover/yr", "score_med_if_pos") else f"{v:.3f}"))
    d = d.rename(columns=HDR)
    lines = ["| " + " | ".join(d.columns) + " |", "|" + "|".join(["---"] * len(d.columns)) + "|"]
    lines += ["| " + " | ".join(str(v).replace(" | ", " + ") for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ens")
    ap.add_argument("--rank", default="ret_med", choices=list(RANKINGS))
    args = ap.parse_args()
    df = pd.read_csv(RESULTS / f"research_tilts_{args.out}.csv")
    bs = pd.read_csv(RESULTS / f"research_bootstrap_{args.out}.csv")
    keys = RANKINGS[args.rank]
    fams = [f for f in FAMILY_LABEL if f != "bench" and (df.family == f).any()]
    out = []
    out.append(f"### Best config per family, chosen within each split (ranking: {' > '.join(keys)})\n")
    for split in SPLITS:
        rows = [pick(df, split, f, keys) for f in fams] + [r for _, r in df[(df.family == "bench") & (df.split == split)].iterrows()]
        t = pd.DataFrame(rows)
        t = t.assign(Family=t["family"].map(lambda f: FAMILY_LABEL.get(f, f)), Config=t["name"])
        out.append(f"**{split}**\n\n" + md(t, ["Family", "Config"] + COLS) + "\n")
    out.append("### Val-selected configs carried to test and holdout\n")
    sel = {f: pick(df, "val", f, keys)["name"] for f in fams}
    for split in SPLITS:
        t = pd.DataFrame([row_of(df, n, split) for n in sel.values()])
        t = t.assign(Family=t["family"].map(lambda f: FAMILY_LABEL.get(f, f)), Config=t["name"])
        out.append(f"**{split}**\n\n" + md(t, ["Family", "Config"] + COLS) + "\n")
    out.append("### Paired block bootstrap (candidate minus pure core; 1000 resamples, 14-day blocks, 95% CI)\n")
    b = bs[bs["rank"] == args.rank].copy()
    b["family"] = b["family"].map(lambda f: FAMILY_LABEL.get(f, f))
    b["d_ret_med [CI]"] = b.apply(lambda r: f"{r.d_ret_med:+.4f} [{r.d_ret_lo:+.4f}, {r.d_ret_hi:+.4f}]", axis=1)
    b["P(d_ret>0)"] = b.p_ret_gt.map(lambda v: f"{v:.2f}")
    b["d_p_pos [CI]"] = b.apply(lambda r: f"{r.d_p_pos:+.3f} [{r.d_pos_lo:+.3f}, {r.d_pos_hi:+.3f}]", axis=1)
    b["P(d_pos>0)"] = b.p_pos_gt.map(lambda v: f"{v:.2f}")
    for view in ("in_split", "val_selected"):
        out.append(f"**{view}**\n\n" + md(b[b.view == view], ["split", "family", "d_ret_med [CI]", "P(d_ret>0)", "d_p_pos [CI]", "P(d_pos>0)"]) + "\n")
    out.append("### Family medians across the whole grid\n")
    med = df[df.family != "bench"].groupby(["family", "split"])[["ret_med", "p_pos", "p_gt3", "maxdd_med", "return", "turnover/yr"]].median().reset_index()
    med["family"] = med["family"].map(lambda f: FAMILY_LABEL.get(f, f))
    out.append(md(med, ["family", "split", "ret_med", "p_pos", "p_gt3", "maxdd_med", "return", "turnover/yr"]) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
