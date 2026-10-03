"""Out-of-sample evaluation.

1. Predict with the saved model on val / test / holdout.
2. Choose portfolio parameters on VAL only (grid), report chosen config on TEST
   (2023-2025) and HOLDOUT (2026 YTD) alongside BTC buy-and-hold and the rule-based
   momentum baseline. Nothing in test/holdout influences any choice.

Validation objective (--objective, default `w14`):
  The competition is scored on ONE 14-day window with an unknown start, and the first
  gate is raw return (top-20 by return per region; only then 0.4 Sortino + 0.3 Sharpe
  + 0.3 Calmar). Full-period statistics over a year say little about that. So each
  config is scored on the distribution of its daily-rolling 14-day windows
  (bot/ml/windows.summarize):
      require  w14_p_pos > 0.5                (more windows up than down, else we
                                               more likely than not miss the gate)
      rank by  pctrank(w14_ret_med) + W14_SCORE_WEIGHT * pctrank(w14_score_med_if_pos)
  i.e. median 14-day return first, with the composite ratio score of the positive
  windows as a secondary criterion. Percentile ranks are used because the composite
  score on 14-day windows (annualised Calmar) spans 10-80 and would otherwise swamp
  the return term (that is exactly what an additive 0.02 weight did in the first run).
  `--objective full` keeps the legacy rule: full-period composite score among configs
  with positive full-period return.

    python -m bot.ml.evaluate --tag base
    python -m bot.ml.evaluate --tag cnn64 --fee-scenario   # + taker / 50% maker / 100% maker table
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bot.backtest import metrics
from bot.backtest import run as momentum_run
from bot.strategy import Params as MomParams
from bot.ml.dataset import SPLITS, WindowDataset, get_panel, sample_index
from bot.ml.features import HORIZONS
from bot.ml.model import CNN
from bot.ml.portfolio import MEDIAN_SPREAD_BPS, PortParams, report, simulate
from bot.ml.train import DEVICE, MODELS, predict, rank_ic
from bot.ml.windows import SUMMARY_FIELDS, summarize
from bot.risk import RiskParams

RESULTS = Path(__file__).resolve().parents[2] / "results"

OBJECTIVES = ("w14", "full")
W14_SCORE_WEIGHT = 0.5    # weight of the score percentile rank relative to the return percentile rank
W14_P_POS_GATE = 0.5

# execution assumptions for --fee-scenario: (label, simulate kwargs)
FEE_SCENARIOS = (
    ("taker", dict(maker_frac=0.0)),                                        # = main report
    ("taker+spread", dict(maker_frac=0.0, spread_bps=MEDIAN_SPREAD_BPS)),
    ("maker50+spread", dict(maker_frac=0.5, spread_bps=MEDIAN_SPREAD_BPS)),
    ("maker100", dict(maker_frac=1.0, spread_bps=MEDIAN_SPREAD_BPS)),       # spread only hits the taker fraction
)

# columns printed in the two result tables (the CSV keeps everything)
FULL_COLS = ["split", "name", "return", "sharpe", "sortino", "calmar", "max_dd", "score", "trades",
             "win14_med_ret", "win14_med_score", "win14_pos"]
W14_COLS = ["split", "name", "w14_n_windows", "w14_ret_med", "w14_ret_p10", "w14_ret_p90", "w14_p_pos",
            "w14_p_gt3", "w14_p_gt5", "w14_maxdd_med", "w14_maxdd_p10", "w14_score_med", "w14_score_med_if_pos"]


def load_model(tag: str):
    ck = torch.load(MODELS / f"{tag}.pt", map_location="cpu", weights_only=False)
    m = CNN(hidden=ck["hidden"], drop=ck["drop"])
    m.load_state_dict(ck["state"])
    return m.to(DEVICE), ck


def predictions(tag: str, split: str, feat, valid, times, coins) -> dict[int, pd.DataFrame]:
    """{horizon: DataFrame [hours x coins]} of predicted vol-normalised returns (NaN where not predictable)."""
    caches = {h: RESULTS / f"pred_{tag}_{split}_{h}.parquet" for h in HORIZONS}
    if all(c.exists() for c in caches.values()):
        return {h: pd.read_parquet(c) for h, c in caches.items()}
    model, _ = load_model(tag)
    idx = sample_index(valid, times, split, None)
    pred = predict(model, WindowDataset(feat, None, idx))
    s, e = (pd.Timestamp(x, tz="UTC") for x in SPLITS[split])
    tm = (times >= s) & (times < e)
    RESULTS.mkdir(exist_ok=True)
    out = {}
    for k, h in enumerate(HORIZONS):
        arr = np.full((len(times), len(coins)), np.nan, np.float32)
        arr[idx[:, 0], idx[:, 1]] = pred[:, k]
        out[h] = pd.DataFrame(arr[tm], index=times[tm], columns=coins)
        out[h].to_parquet(caches[h])
    return out


def frames_for(split: str, pred: pd.DataFrame, vol, lc, feat, times, coins):
    t = pred.index
    sel = times.get_indexer(t)
    volf = pd.DataFrame(vol[sel], index=t, columns=coins)
    lcf = pd.DataFrame(lc[sel], index=t, columns=coins)
    mkt = pd.Series(feat[sel, 0, 7], index=t)
    return volf, lcf, mkt


def grid(short_k: int = 0):
    """Portfolio parameter grid. Long-only by default; short_k>0 adds a short book."""
    for head, top_k, tv, rebal, band, thresh, regime in itertools.product(
            HORIZONS, (3, 5, 8), (0.3, 0.5, 0.8), (12, 24, 48), (0.05, 0.10), (0.0, 0.1), (False, True)):
        yield PortParams(head=head, top_k=top_k, short_k=short_k, target_vol=tv, rebal_h=rebal, band=band,
                         thresh=thresh, short_thresh=-thresh, use_regime=regime)


GRID_KEYS = list(PortParams().__dict__)
GRID_FIELDS = ["ret", "score", "maxdd", "trades"] + [f"w14_{k}" for k in SUMMARY_FIELDS]


def load_val_grid(tag: str, configs: list[PortParams]) -> tuple[pd.DataFrame | None, Path | None]:
    """Reuse a cached validation grid if it covers every config with the current fields:
    results/valgrid_<tag>.csv (this script) or results/grid_<tag>_val.csv (bot/ml/stability.py
    runs the identical simulation). Returns (frame with a `p` column, source) or (None, None)."""
    cf = pd.DataFrame([q.__dict__ for q in configs])
    for f in (RESULTS / f"valgrid_{tag}.csv", RESULTS / f"grid_{tag}_val.csv"):
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if not set(GRID_KEYS + GRID_FIELDS) <= set(df.columns):
            continue
        m = cf.merge(df[GRID_KEYS + GRID_FIELDS].drop_duplicates(GRID_KEYS), on=GRID_KEYS, how="left")
        if len(m) == len(cf) and m["ret"].notna().all():
            m["p"] = configs
            return m, f
    return None, None


def w14_objective(ret_med, score_med_if_pos):
    """Rolling-14d ranking value: percentile rank of the median window return plus
    W14_SCORE_WEIGHT times the percentile rank of the median composite score of the
    positive windows (NaN score, i.e. no positive window, ranks lowest). Ranks are
    computed within the arrays given, so the value is only comparable within one call."""
    r = pd.Series(ret_med, dtype=float).rank(pct=True)
    sc = pd.Series(score_med_if_pos, dtype=float).fillna(-np.inf).rank(pct=True)
    return (r + W14_SCORE_WEIGHT * sc).to_numpy()


def objective(vr: pd.DataFrame, kind: str = "w14") -> tuple[pd.Series, pd.Series]:
    """Return (objective value, gate passed) per grid row. Gated-out rows are pushed to the
    bottom but keep their relative order so a best-effort choice always exists."""
    if kind == "full":
        gate = vr["ret"] > 0
        obj = vr["score"].astype(float)
    elif kind == "w14":
        gate = vr["w14_p_pos"] > W14_P_POS_GATE
        obj = pd.Series(w14_objective(vr["w14_ret_med"], vr["w14_score_med_if_pos"]), index=vr.index)
    else:
        raise ValueError(f"unknown objective {kind!r}; choose from {OBJECTIVES}")
    return pd.Series(np.where(gate, obj, obj - 1e9), index=vr.index), gate


def _val_score(args):
    p, pred, volf, lcf, mkt = args
    eq, tr = simulate(pred, volf, lcf, p, mkt=mkt)
    m = metrics(eq)
    w14 = summarize(eq)
    return {"p": p, "ret": m["return"], "score": m["score"], "maxdd": m["max_dd"], "trades": len(tr),
            **{f"w14_{k}": v for k, v in w14.items()}}


def pool_map(fn, jobs, workers: int, label: str, chunksize: int = 8) -> list:
    """ProcessPoolExecutor.map with a progress line every ~10% (grids run for a long time)."""
    from concurrent.futures import ProcessPoolExecutor
    import time
    out, t0, n = [], time.time(), len(jobs)
    marks = {max(1, round(n * k / 10)) for k in range(1, 11)}
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(fn, jobs, chunksize=chunksize), 1):
            out.append(r)
            if i in marks:
                el = time.time() - t0
                print(f"  {label}: {i}/{n} configs  {el / 60:.1f} min elapsed, ~{el / i * (n - i) / 60:.1f} min left", flush=True)
    return out


def _fmt_cfg(r) -> str:
    return (f"ret {r.ret:+.3f} score {r.score:6.2f} maxdd {r.maxdd:.3f} trades {int(r.trades):4d} | "
            f"w14 ret_med {r.w14_ret_med:+.4f} p_pos {r.w14_p_pos:.2f} p10 {r.w14_ret_p10:+.4f} "
            f"score_if_pos {r.w14_score_med_if_pos:5.2f}")


def fee_scenarios(configs: dict[str, PortParams], frames: dict, spread_bps: float = MEDIAN_SPREAD_BPS) -> pd.DataFrame:
    """Chosen configs re-simulated under each execution assumption in FEE_SCENARIOS."""
    rows = []
    for split, (preds, volf, lcf, mkt) in frames.items():
        years = max((lcf.index[-1] - lcf.index[0]).days / 365, 1e-6)
        for name, p in configs.items():
            for label, kw in FEE_SCENARIOS:
                kw = {**kw, **({"spread_bps": spread_bps} if "spread_bps" in kw else {})}
                eq, tr = simulate(preds[p.head], volf, lcf, p, mkt=mkt, **kw)
                rep = report(name, eq)
                rows.append({"split": split, "name": name, "fill": label, "return": rep["return"],
                             "score": rep["score"], "max_dd": rep["max_dd"],
                             "w14_ret_med": rep.get("w14_ret_med"), "w14_p_pos": rep.get("w14_p_pos"),
                             "w14_score_med_if_pos": rep.get("w14_score_med_if_pos"),
                             "fee_cost/yr": round(float(tr.fee_cost.sum()) / years, 4) if len(tr) else 0.0,
                             "spread_cost/yr": round(float(tr.spread_cost.sum()) / years, 4) if len(tr) else 0.0})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--objective", choices=OBJECTIVES, default="w14",
                    help="validation ranking: w14 = rolling 14d windows (default), full = legacy full-period score")
    ap.add_argument("--regrid", action="store_true",
                    help="recompute the validation grid even if results/valgrid_<tag>.csv or grid_<tag>_val.csv is fresh")
    ap.add_argument("--fee-scenario", action="store_true",
                    help="also report the chosen configs under taker-only / 50%% maker / 100%% maker fills")
    ap.add_argument("--spread-bps", type=float, default=MEDIAN_SPREAD_BPS,
                    help="half-spread (bps) charged on taker notional in the fee-scenario table")
    args = ap.parse_args()
    feat, vol, lc, valid, times, coins = get_panel()

    rows = []
    frames = {}
    for split in ("val", "test", "holdout"):
        preds = predictions(args.tag, split, feat, valid, times, coins)
        volf, lcf, mkt = frames_for(split, preds[HORIZONS[0]], vol, lc, feat, times, coins)
        frames[split] = (preds, volf, lcf, mkt)
        # information coefficient of each head on this split (hourly cross-sectional rank IC)
        for h in HORIZONS:
            fwd = (lcf.shift(-h) - lcf) / (volf * np.sqrt(h))
            ic = pd.concat([preds[h].stack(), fwd.stack()], axis=1).dropna()
            ic.columns = ["p", "y"]
            ic_t = ic.groupby(level=0).apply(lambda g: g.p.corr(g.y, method="spearman") if len(g) >= 5 else np.nan)
            # overlapping targets: t-stat uses non-overlapping count
            n_eff = ic_t.notna().sum() / h
            print(f"{split}: head {h:3d}h  rank IC mean {ic_t.mean():+.4f}  t-stat(non-overlap) "
                  f"{ic_t.mean() / ic_t.std() * np.sqrt(n_eff):+.1f}  coverage {preds[h].notna().mean().mean():.0%}", flush=True)

    # --- select portfolio params on validation only ---
    preds, volf, lcf, mkt = frames["val"]
    configs = list(grid(0)) + list(grid(2))
    RESULTS.mkdir(exist_ok=True)
    vr, src = (None, None) if args.regrid else load_val_grid(args.tag, configs)
    if vr is None:
        jobs = [(p, preds[p.head], volf, lcf, mkt) for p in configs]
        vr = pd.DataFrame(pool_map(_val_score, jobs, args.workers, "val grid"))
        pd.concat([vr.drop(columns="p"), pd.DataFrame([q.__dict__ for q in vr.p])], axis=1).to_csv(
            RESULTS / f"valgrid_{args.tag}.csv", index=False)
    else:
        print(f"\nreusing cached validation grid {src.name} ({len(vr)} configs; --regrid recomputes)")
    vr["obj"], vr["gate"] = objective(vr, args.objective)
    vr = vr.sort_values("obj", ascending=False)
    print(f"\nObjective `{args.objective}`: {int(vr.gate.sum())} of {len(vr)} configs pass the gate "
          f"({'w14 p_pos > 0.5' if args.objective == 'w14' else 'full-period return > 0'}).")
    print("Top validation configs:")
    for _, r in vr.head(5).iterrows():
        print(f"  {_fmt_cfg(r)}  {r.p}")
    is_lo = vr.p.apply(lambda q: q.short_k == 0)
    best_long_only = vr[is_lo].iloc[0]
    best_ls = vr[~is_lo].iloc[0]
    for lbl, r in (("long-only", best_long_only), ("long/short", best_ls)):
        if not r.gate:
            print(f"  WARNING: no {lbl} config passes the gate; best-effort choice reported")
    best_long_only, best_ls = best_long_only.p, best_ls.p
    print(f"\nchosen long-only: {best_long_only}\nchosen long/short: {best_ls}")

    # --- report on all splits with the frozen configs ---
    print("\n=== Frozen configs on each split ===")
    out = []
    for split in ("val", "test", "holdout"):
        preds, volf, lcf, mkt = frames[split]
        for name, p in (("cnn_long_only", best_long_only), ("cnn_long_short", best_ls)):
            eq, tr = simulate(preds[p.head], volf, lcf, p, mkt=mkt)
            eq.to_csv(RESULTS / f"equity_{args.tag}_{name}_{split}.csv")
            out.append({"split": split, **report(name, eq), "trades": len(tr)})
        btc = np.exp(lcf["BTC"]).dropna()
        out.append({"split": split, **report("btc_hold", btc / btc.iloc[0]), "trades": 0})
        # rule-based momentum baseline (bot/strategy.py) on the same hours; needs warm-up history
        s0 = times.get_loc(lcf.index[0])
        warm = max(0, s0 - 24 * 40)
        closes = pd.DataFrame(np.exp(lc[warm: s0 + len(lcf)]), index=times[warm: s0 + len(lcf)],
                              columns=[c + "/USD" for c in coins])
        mom_eq, mom_tr = momentum_run(closes, MomParams(lookbacks=(72, 168, 336), trend_ema=168, regime_ema=336),
                                      rebalance_band=0.06, warmup=s0 - warm)
        out.append({"split": split, **report("momentum_rules", mom_eq), "trades": len(mom_tr)})
        eqw = np.exp(lcf).ffill().pct_change(fill_method=None).mean(axis=1).fillna(0).add(1).cumprod()
        out.append({"split": split, **report("equal_weight_all", eqw), "trades": 0})
    df = pd.DataFrame(out)
    pd.set_option("display.width", 250)
    print("-- full period --")
    print(df[[c for c in FULL_COLS if c in df.columns]].to_string(index=False))
    print("-- rolling 14-day windows (daily step; the competition objective) --")
    print(df[[c for c in W14_COLS if c in df.columns]].to_string(index=False))
    df.to_csv(RESULTS / f"summary_{args.tag}.csv", index=False)
    (RESULTS / f"config_{args.tag}.json").write_text(json.dumps(
        {"best": best_long_only.__dict__, "long_short": best_ls.__dict__, "objective": args.objective}, indent=1))

    if args.fee_scenario:
        print(f"\n=== Execution assumptions for the chosen configs (half-spread {args.spread_bps} bps on taker notional) ===")
        fs = fee_scenarios({"cnn_long_only": best_long_only, "cnn_long_short": best_ls}, frames, args.spread_bps)
        print(fs.to_string(index=False))
        fs.to_csv(RESULTS / f"feescenario_{args.tag}.csv", index=False)


if __name__ == "__main__":
    main()
