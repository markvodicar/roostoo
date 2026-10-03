"""Selection-stability analysis: does doing well on the validation year predict
doing well on the test years?

Runs the whole portfolio grid on BOTH val and test, then reports
  * Spearman correlation across configs between val metrics and test metrics
  * what several selection rules applied on val would have delivered on test
    (top-1 by full-year score, top-1 by median 14-day-window score, top-decile
    average, long-only variants, and a fixed "no-selection" sensible default)
This is an analysis of the *selection procedure*; it is test-informed by design
and is reported as such.

    python -m bot.ml.stability --tag cnn64
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from bot.backtest import metrics, window_report
from bot.ml.dataset import get_panel
from bot.ml.evaluate import RESULTS, frames_for, grid, pool_map, predictions
from bot.ml.features import HORIZONS
from bot.ml.portfolio import PortParams, simulate
from bot.ml.windows import summarize


def _run(args):
    p, pred, volf, lcf, mkt = args
    eq, tr = simulate(pred, volf, lcf, p, mkt=mkt)
    m = metrics(eq)
    wr = window_report(eq)
    w14 = summarize(eq)  # rolling daily-step 14d windows (the evaluate.py objective)
    return {**p.__dict__, "ret": m["return"], "score": m["score"], "maxdd": m["max_dd"], "sharpe": m["sharpe"],
            "win14_med_score": wr["score"].median(), "win14_med_ret": wr["return"].median(),
            "win14_pos": (wr["return"] > 0).mean(), "trades": len(tr),
            **{f"w14_{k}": v for k, v in w14.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cnn64")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--regen", action="store_true",
                    help="ignore cached results/grid_*.csv and re-run the grid (stale caches without w14_* fields are redone anyway)")
    args = ap.parse_args()
    feat, vol, lc, valid, times, coins = get_panel()
    configs = list(grid(0)) + list(grid(2))
    keys = list(PortParams().__dict__)
    res = {}
    for split in ("val", "test"):
        f = RESULTS / f"grid_{args.tag}_{split}.csv"
        if f.exists() and not args.regen:
            cached = pd.read_csv(f)
            if "w14_ret_med" in cached.columns:  # written with the current fee schedule and window fields
                res[split] = cached
                continue
            print(f"{split}: cached grid predates the w14_* fields (old fee schedule); regenerating", flush=True)
        preds = predictions(args.tag, split, feat, valid, times, coins)
        volf, lcf, mkt = frames_for(split, preds[HORIZONS[0]], vol, lc, feat, times, coins)
        jobs = [(p, preds[p.head], volf, lcf, mkt) for p in configs]
        rows = pool_map(_run, jobs, args.workers, f"{split} grid")
        res[split] = pd.DataFrame(rows)
        res[split].to_csv(f, index=False)
        print(f"{split}: grid done ({len(rows)} configs)", flush=True)

    v, t = res["val"], res["test"]
    both = v.merge(t, on=keys, suffixes=("_val", "_test"))
    print(f"\n{len(both)} configs. Spearman(val -> test) across configs:")
    for m in ("ret", "score", "win14_med_score", "sharpe"):
        print(f"  {m:16s} {both[m + '_val'].corr(both[m + '_test'], method='spearman'):+.3f}")
    lo = both[both.short_k == 0]
    print(f"long-only subset ({len(lo)}): score corr {lo.score_val.corr(lo.score_test, method='spearman'):+.3f}, "
          f"ret corr {lo.ret_val.corr(lo.ret_test, method='spearman'):+.3f}")

    def show(name, df):
        print(f"  {name:42s} n={len(df):4d}  TEST ret {df.ret_test.mean():+.3f}  score {df.score_test.mean():+.2f}  "
              f"maxdd {df.maxdd_test.mean():.3f}  win14 med score {df.win14_med_score_test.mean():+.2f}")

    print("\nSelection rules applied on VAL -> realised on TEST (means over selected configs):")
    pos = both[both.ret_val > 0]
    show("top-1 by val score (evaluate.py --objective full)", pos.nlargest(1, "score_val"))
    if "w14_ret_med_val" in both.columns:  # grid CSVs written after the rolling-window fields were added
        from bot.ml.evaluate import w14_objective
        gated = both[both.w14_p_pos_val > 0.5].copy()
        gated["obj"] = w14_objective(gated.w14_ret_med_val, gated.w14_score_med_if_pos_val)
        show("top-1 by val w14 objective (evaluate.py default)", gated.nlargest(1, "obj"))
        show("top-10% by val w14 objective", gated.nlargest(max(1, len(gated) // 10), "obj"))
        show("top-1 by val w14 ret_med only (gate, no score term)", gated.nlargest(1, "w14_ret_med_val"))
        glo = gated[gated.short_k == 0]
        show("long-only top-1 by val w14 objective", glo.nlargest(1, "obj"))
        show("long-only top-10% by val w14 objective", glo.nlargest(max(1, len(glo) // 10), "obj"))
        show("long-only top-1 by val w14 ret_med only", glo.nlargest(1, "w14_ret_med_val"))
        print(f"  (w14 objective: {len(gated)} of {len(both)} configs pass the p_pos>0.5 gate on val; "
              f"Spearman val->test w14_ret_med {both.w14_ret_med_val.corr(both.w14_ret_med_test, method='spearman'):+.3f})")
    else:
        print("  (grid CSVs predate the w14_* fields; re-run with --regen to score the w14 objective rule)")
    show("top-1 by val median 14d-window score", pos.nlargest(1, "win14_med_score_val"))
    show("top-10% by val score", pos.nlargest(max(1, len(pos) // 10), "score_val"))
    show("top-10% by val win14 score", pos.nlargest(max(1, len(pos) // 10), "win14_med_score_val"))
    lo_pos = lo[lo.ret_val > 0]
    show("long-only top-1 by val score", lo_pos.nlargest(1, "score_val"))
    show("long-only top-10% by val score", lo_pos.nlargest(max(1, len(lo_pos) // 10), "score_val"))
    show("ALL long-only configs (no selection)", lo)
    show("ALL long/short configs (no selection)", both[both.short_k > 0])
    for h in HORIZONS:
        show(f"long-only, head={h}h, all configs", lo[lo["head"] == h])  # lo.head is DataFrame.head
    for rb in (12, 24, 48):
        show(f"long-only, rebal={rb}h, all configs", lo[lo.rebal_h == rb])
    for tv in (0.3, 0.5, 0.8):
        show(f"long-only, target_vol={tv}, all configs", lo[lo.target_vol == tv])
    print("\nBest TEST configs (hindsight, for reference only):")
    for _, r in both.nlargest(5, "score_test").iterrows():
        print(f"  TEST ret {r.ret_test:+.3f} score {r.score_test:5.2f} | VAL ret {r.ret_val:+.3f} score {r.score_val:5.2f} | "
              f"head {r['head']} k{r.top_k}/s{r.short_k} tv{r.target_vol} rb{r.rebal_h} band{r.band} th{r.thresh} reg{r.use_regime}")


if __name__ == "__main__":
    main()
