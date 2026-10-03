"""Pure core vs core + tilt on rolling 14-day windows (val, test, holdout).

Hypothesis under test (PLAN.md, workstream B): does a 30% CNN tilt improve the
distribution of 14-day window returns over the best pure BTC/ETH core? Alternatives
at the same budget: cross-sectional momentum, per-asset trend following, an
extended core (BTC/ETH/SOL + top-3 by dollar volume) and a BTC short overlay.

Decision rule: the CNN tilt is kept only if it improves ret_med AND p_pos (or
score_med_if_pos without hurting ret_med) over the best pure core on BOTH val and
test. Holdout (2026 YTD) is reported last and never used to choose.

    python -m bot.research.compare_tilts --tags ens            # ~900 configs x 3 splits
    python -m bot.research.compare_tilts --tags ens ens24 --out ens24
Outputs results/research_tilts[_out].csv, results/research_bootstrap[_out].csv,
results/research_equity[_out].parquet (daily equity of every config) and a printed verdict.
"""
from __future__ import annotations

import argparse
import itertools
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from bot.backtest import metrics
from bot.fees import TAKER
from bot.ml.dataset import get_panel
from bot.ml.evaluate import RESULTS, frames_for, predictions
from bot.ml.portfolio import simulate
from bot.ml.windows import summarize as slow_summarize
from bot.research import fastwin
from bot.research.bootstrap import paired_bootstrap
from bot.research.tilts import CoreTiltWeights, TiltConfig, build_indicators, slice_indicators

SPLITS = ("val", "test", "holdout")
SCRATCH = Path("/private/tmp/claude-501/-Users-izovoha-roostoo-bot/4bf06f6b-5307-46cd-a157-355635af6074/scratchpad")
PRIMARY = ["ret_med", "ret_p10", "p_pos", "p_gt3", "maxdd_med", "score_med_if_pos", "return", "turnover/yr"]
RANKINGS = {  # lexicographic "best" within a split
    "ret_med": ["ret_med", "p_pos", "score_med_if_pos"],          # PLAN.md primary objective
    "p_gt3": ["p_gt3", "ret_med", "p_pos"],                       # competition-gate view: P(14d return > 3%)
}
RANK_KEYS = RANKINGS["ret_med"]
FAMILY_LABEL = {"core": "pure core", "cnn": "core+CNN", "xs_mom": "core+xs-mom", "trend": "core+trend",
                "ext_core": "extended core", "short_core": "core w/ BTC short", "bench": "benchmark"}


# ----------------------------------------------------------------------------- grid
def build_grid(tags: list[str], quick: bool = False) -> list[TiltConfig]:
    spans, cvs, rebals, bands = (10, 20, 30, 50), (0.3, 0.5, 0.8, 1.0), (24, 48, 72), (0.02, 0.05)
    base_spans, base_cvs, base_rebals = (20, 30, 50), (0.5, 1.0), (24, 48)
    core_fracs, top_ks, heads, moms = (0.5, 0.7, 0.85), (3, 5, 8), (24, 336), (168, 336, 720)
    if quick:
        spans, cvs, rebals, bands = (20, 50), (0.5, 1.0), (48,), (0.02,)
        base_spans, base_cvs, base_rebals = (20,), (0.5,), (48,)
        core_fracs, top_ks, heads, moms = (0.7,), (3, 5), (24,), (336,)
    g: list[TiltConfig] = []
    for s, cv, rb, b in itertools.product(spans, cvs, rebals, bands):
        g.append(TiltConfig("core", 1.0, s, cv, rb, b))
        g.append(TiltConfig("short_core", 1.0, s, cv, rb, b, short_core=True))
        if b == 0.02:
            g.append(TiltConfig("ext_core", 1.0, s, cv, rb, b, ext_core=True))
    for s, cv, rb, cf in itertools.product(base_spans, base_cvs, base_rebals, core_fracs):
        for tag, h, k in itertools.product(tags, heads, top_ks):
            g.append(TiltConfig("cnn", cf, s, cv, rb, 0.02, tilt="cnn", head=h, top_k=k, pred_tag=tag))
        for L, k in itertools.product(moms, top_ks):
            g.append(TiltConfig("xs_mom", cf, s, cv, rb, 0.02, tilt="xs_mom", mom_h=L, top_k=k))
        for k in top_ks:
            g.append(TiltConfig("trend", cf, s, cv, rb, 0.02, tilt="trend", top_k=k))
    return g


# ----------------------------------------------------------------------------- data prep
def prepare(tags: list[str]) -> dict:
    feat, vol, lc, valid, times, coins = get_panel()
    coins = [str(c) for c in coins]
    ind = build_indicators(lc, vol, valid, times, coins)
    splits = {}
    for split in SPLITS:
        preds = {tag: predictions(tag, split, feat, valid, times, coins) for tag in tags}
        ref = preds[tags[0]][24]
        volf, lcf, _ = frames_for(split, ref, vol, lc, feat, times, coins)
        rows = times.get_indexer(ref.index)
        splits[split] = {"rows": rows, "volf": volf, "lcf": pd.DataFrame(ind["lc"][rows], index=ref.index, columns=coins),
                         "preds": {(tag, h): preds[tag][h] for tag in tags for h in (24, 336)},
                         "ind": slice_indicators(ind, rows)}
    return {"splits": splits, "coins": coins}


_D: dict = {}


def _init(path: str):
    global _D
    with open(path, "rb") as f:
        _D = pickle.load(f)


def run_config(cfg: TiltConfig) -> tuple[list[dict], list[pd.Series]]:
    rows, curves = [], []
    for split in SPLITS:
        d = _D["splits"][split]
        pred = d["preds"][(cfg.pred_tag, cfg.head)] if cfg.tilt == "cnn" else d["preds"][next(iter(d["preds"]))]
        wf = CoreTiltWeights(cfg, d["ind"])
        eq, tr = simulate(pred, d["volf"], d["lcf"], cfg.port_params(), weight_fn=wf, fee=TAKER)
        years = max((eq.index[-1] - eq.index[0]).days / 365, 0.1)
        s = fastwin.summarize(eq) if _D.get('fast', True) else slow_summarize(eq)
        m = metrics(eq)
        rows.append({"name": cfg.name, "family": cfg.family, "split": split, **cfg.to_dict(),
                     **{k: s.get(k, np.nan) for k in ("n_windows", "ret_med", "ret_p10", "ret_p90", "p_pos", "p_gt3",
                                                      "maxdd_med", "score_med", "score_med_if_pos")},
                     "return": m["return"], "max_dd": m["max_dd"], "score": m["score"],
                     "turnover/yr": (tr.turnover.sum() / years) if len(tr) else 0.0, "trades": len(tr)})
        dq = eq.resample("1D").last().dropna()
        curves.append(pd.Series(dq.values, index=pd.MultiIndex.from_arrays(
            [[cfg.name] * len(dq), [split] * len(dq), dq.index], names=["name", "split", "t"])))
    return rows, curves


def benchmarks(tag: str) -> list[tuple[list[dict], list[pd.Series]]]:
    """Buy-and-hold references (no fees): BTC, ETH, 50/50 BTC-ETH (daily rebalanced, fee-free)."""
    out = []
    for name, fn in (("btc_hold", lambda px: px["BTC"]), ("eth_hold", lambda px: px["ETH"]),
                     ("btc_eth_5050", lambda px: (1 + px[["BTC", "ETH"]].pct_change().fillna(0).mean(axis=1)).cumprod())):
        rows, curves = [], []
        for split in SPLITS:
            d = _D["splits"][split]
            px = np.exp(d["lcf"][["BTC", "ETH"]])
            eq = fn(px)
            eq = eq / eq.iloc[0]
            s, m = fastwin.summarize(eq), metrics(eq)
            cfg = TiltConfig("bench")
            rows.append({"name": name, "family": "bench", "split": split, **cfg.to_dict(),
                         **{k: s.get(k, np.nan) for k in ("n_windows", "ret_med", "ret_p10", "ret_p90", "p_pos", "p_gt3",
                                                          "maxdd_med", "score_med", "score_med_if_pos")},
                         "return": m["return"], "max_dd": m["max_dd"], "score": m["score"], "turnover/yr": 0.0, "trades": 0})
            dq = eq.resample("1D").last().dropna()
            curves.append(pd.Series(dq.values, index=pd.MultiIndex.from_arrays(
                [[name] * len(dq), [split] * len(dq), dq.index], names=["name", "split", "t"])))
        out.append((rows, curves))
    return out


# ----------------------------------------------------------------------------- selection / verdict
def best_in(df: pd.DataFrame, keys: list[str] = RANK_KEYS) -> pd.Series:
    return df.sort_values(keys, ascending=False).iloc[0]


def pick(df: pd.DataFrame, split: str, family: str, keys: list[str] = RANK_KEYS) -> pd.Series:
    return best_in(df[(df.split == split) & (df.family == family)], keys)


def row_of(df: pd.DataFrame, name: str, split: str) -> pd.Series:
    return df[(df.name == name) & (df.split == split)].iloc[0]


def beats(c: pd.Series, r: pd.Series, eps: float = 1e-6) -> tuple[bool, str]:
    """PLAN.md rule: improves ret_med AND p_pos, or score_med_if_pos without hurting ret_med."""
    a = c.ret_med > r.ret_med + eps and c.p_pos > r.p_pos + eps
    b = c.score_med_if_pos > r.score_med_if_pos + eps and c.ret_med >= r.ret_med - eps
    why = []
    if a:
        why.append("ret_med & p_pos up")
    if b:
        why.append("score_med_if_pos up, ret_med not worse")
    return (a or b), (", ".join(why) or "no improvement")


def fmt(r: pd.Series) -> str:
    return (f"ret_med {r.ret_med:+.4f} p10 {r.ret_p10:+.3f} p_pos {r.p_pos:.2f} p>3% {r.p_gt3:.2f} "
            f"dd_med {r.maxdd_med:.3f} sc+ {r.score_med_if_pos:6.1f} | full {r['return']:+.3f} to/yr {r['turnover/yr']:.0f}")


def verdict(df: pd.DataFrame, eqs: pd.Series, n_boot: int, families: list[str], rank: str = "ret_med") -> tuple[list[dict], str]:
    keys = RANKINGS[rank]
    lines, boots = [], []
    lines.append("\n" + "#" * 100 + f"\n#  RANKING: {rank} first ({' > '.join(keys)})\n" + "#" * 100)
    lines.append("\n" + "=" * 100 + "\nVIEW 1: best config of each family chosen WITHIN each split (in-sample on that split)")
    in_split: dict[str, dict[str, pd.Series]] = {}
    for split in SPLITS:
        in_split[split] = {f: pick(df, split, f, keys) for f in families if ((df.split == split) & (df.family == f)).any()}
        ref = in_split[split]["core"]
        lines.append(f"\n[{split}] best pure core: {ref['name']}\n        {fmt(ref)}")
        for f, c in in_split[split].items():
            if f == "core":
                continue
            ok, why = beats(c, ref)
            lines.append(f"  {FAMILY_LABEL[f]:<18s} {c['name']}\n        {fmt(c)}\n        -> {'BEATS' if ok else 'does not beat'} core ({why})")
            bs = paired_bootstrap(eqs.xs((ref['name'], split), level=[0, 1]), eqs.xs((c['name'], split), level=[0, 1]), n_boot=n_boot)
            boots.append({"rank": rank, "view": "in_split", "split": split, "family": f, "ref": ref["name"], "cand": c["name"], **bs})
            lines.append(f"        bootstrap d(ret_med) {bs['d_ret_med']:+.4f} [{bs['d_ret_lo']:+.4f},{bs['d_ret_hi']:+.4f}] P(>0)={bs['p_ret_gt']:.2f}"
                         f"   d(p_pos) {bs['d_p_pos']:+.3f} [{bs['d_pos_lo']:+.3f},{bs['d_pos_hi']:+.3f}] P(>0)={bs['p_pos_gt']:.2f}")

    lines.append("\n" + "=" * 100 + "\nVIEW 2: configs selected on VAL only, carried to test and holdout (honest out-of-sample)")
    sel = in_split["val"]
    for split in SPLITS:
        ref = row_of(df, sel["core"]["name"], split)
        lines.append(f"\n[{split}] val-selected pure core: {ref['name']}\n        {fmt(ref)}")
        for f, cv in sel.items():
            if f == "core":
                continue
            c = row_of(df, cv["name"], split)
            ok, why = beats(c, ref)
            lines.append(f"  {FAMILY_LABEL[f]:<18s} {c['name']}\n        {fmt(c)}\n        -> {'BEATS' if ok else 'does not beat'} core ({why})")
            bs = paired_bootstrap(eqs.xs((ref['name'], split), level=[0, 1]), eqs.xs((c['name'], split), level=[0, 1]), n_boot=n_boot)
            boots.append({"rank": rank, "view": "val_selected", "split": split, "family": f, "ref": ref["name"], "cand": c["name"], **bs})
            lines.append(f"        bootstrap d(ret_med) {bs['d_ret_med']:+.4f} [{bs['d_ret_lo']:+.4f},{bs['d_ret_hi']:+.4f}] P(>0)={bs['p_ret_gt']:.2f}"
                         f"   d(p_pos) {bs['d_p_pos']:+.3f} [{bs['d_pos_lo']:+.3f},{bs['d_pos_hi']:+.3f}] P(>0)={bs['p_pos_gt']:.2f}")

    lines.append("\n" + "=" * 100 + "\nVIEW 3: paired - best CNN tilt per split vs the SAME core base with the tilt budget in cash (cf=1.0, tilt none)")
    for split in SPLITS:
        for f in [x for x in families if x in ("cnn", "xs_mom", "trend")]:
            if f not in in_split[split]:
                continue
            c = in_split[split][f]
            base = TiltConfig("core", 1.0, int(c.ema_span_d), float(c.core_vol), int(c.rebal_h), float(c.band)).name
            if not ((df.name == base) & (df.split == split)).any():
                continue
            r = row_of(df, base, split)
            ok, why = beats(c, r)
            lines.append(f"[{split}] {FAMILY_LABEL[f]:<12s} {c['name']}\n        tilt {fmt(c)}\n        base {fmt(r)}\n        -> {'BEATS' if ok else 'does not beat'} same-core base ({why})")

    # --- formal rule: CNN must beat best pure core on BOTH val and test, in view 1 (and we report view 2) ---
    lines.append("\n" + "=" * 100)
    out = []
    for f in [x for x in families if x != "core"]:
        v1 = all(beats(in_split[s][f], in_split[s]["core"])[0] for s in ("val", "test") if f in in_split[s])
        v2 = all(beats(row_of(df, sel[f]["name"], s), row_of(df, sel["core"]["name"], s))[0] for s in ("val", "test")) if f in sel else False
        out.append((f, v1, v2))
        lines.append(f"RULE  {FAMILY_LABEL[f]:<18s} beats best pure core on val AND test:  in-split {'YES' if v1 else 'NO '}   val-selected {'YES' if v2 else 'NO '}")
    cnn_keep = any(v1 and v2 for f, v1, v2 in out if f == "cnn")
    lines.append(f"\nVERDICT: CNN tilt {'KEPT' if cnn_keep else 'DROPPED'} under the PLAN.md rule "
                 f"({'passes' if cnn_keep else 'fails'} on val and test in both views).")
    return boots, "\n".join(lines)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["ens"], help="prediction tags for the CNN tilt")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="", help="suffix for results files")
    ap.add_argument("--quick", action="store_true", help="small grid for smoke testing")
    ap.add_argument("--reuse", action="store_true", help="skip simulation, re-run selection/bootstrap on saved csv")
    ap.add_argument("--families", nargs="*", default=None, help="only simulate these families (e.g. cnn)")
    ap.add_argument("--merge-from", default="", help="suffix of a previous run whose rows/equity are merged in "
                    "(e.g. ens: reuse its non-CNN rows when only new CNN tags are simulated)")
    args = ap.parse_args()
    suf = f"_{args.out}" if args.out else ""
    f_csv, f_eq, f_bs = RESULTS / f"research_tilts{suf}.csv", RESULTS / f"research_equity{suf}.parquet", RESULTS / f"research_bootstrap{suf}.csv"

    if args.reuse and f_csv.exists():
        df = pd.read_csv(f_csv)
        eqs = pd.read_parquet(f_eq).set_index(["name", "split", "t"])["eq"]
    else:
        t0 = time.time()
        data = prepare(args.tags)
        SCRATCH.mkdir(parents=True, exist_ok=True)
        pk = SCRATCH / f"compare_tilts{suf}.pkl"
        with open(pk, "wb") as f:
            pickle.dump(data, f, protocol=5)
        grid = build_grid(args.tags, args.quick)
        if args.families:
            grid = [g for g in grid if g.family in set(args.families)]
        _init(str(pk))
        for split in SPLITS:   # verify the vectorised window summary against bot.ml.windows.summarize
            d = _D["splits"][split]
            eq, _ = simulate(d["preds"][(args.tags[0], 24)], d["volf"], d["lcf"], grid[0].port_params(),
                             weight_fn=CoreTiltWeights(grid[0], d["ind"]), fee=TAKER)
            fastwin.check(eq)
        print("fast window summary verified against bot.ml.windows.summarize on all splits", flush=True)
        print(f"prepared data in {time.time() - t0:.0f}s; {len(grid)} configs x {len(SPLITS)} splits", flush=True)
        rows, curves = [], []
        t0 = time.time()
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(pk),)) as ex:
            for n, (r, c) in enumerate(ex.map(run_config, grid, chunksize=4), 1):
                rows += r
                curves += c
                if n % 50 == 0 or n == len(grid):
                    print(f"  {n}/{len(grid)} configs  {time.time() - t0:.0f}s", flush=True)
        if not args.families or "bench" in args.families:
            for r, c in benchmarks(args.tags[0]):
                rows += r
                curves += c
        df = pd.DataFrame(rows)
        eqs = pd.concat(curves)
        if args.merge_from:
            old = pd.read_csv(RESULTS / f"research_tilts_{args.merge_from}.csv")
            old_eq = pd.read_parquet(RESULTS / f"research_equity_{args.merge_from}.parquet").set_index(["name", "split", "t"])["eq"]
            keep = ~old.name.isin(set(df.name))
            df = pd.concat([old[keep], df], ignore_index=True)
            eqs = pd.concat([old_eq[~old_eq.index.get_level_values("name").isin(set(eqs.index.get_level_values("name")))], eqs])
            print(f"merged {keep.sum()} rows from research_tilts_{args.merge_from}.csv", flush=True)
        df.to_csv(f_csv, index=False)
        eqs.rename("eq").reset_index().to_parquet(f_eq)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    families = [f for f in FAMILY_LABEL if (df.family == f).any() and f != "bench"]
    print("\n=== best config per family per split (ranked by ret_med, p_pos, score_med_if_pos) ===")
    top = pd.concat([pick(df, s, f).to_frame().T for s in SPLITS for f in families] + [df[df.family == "bench"]])
    print(top[["split", "family", "name"] + PRIMARY].to_string(index=False))

    print("\n=== pure core pivot: ret_med / p_pos by split ===")
    core = df[df.family == "core"]
    piv = core.pivot(index="name", columns="split", values=["ret_med", "p_pos", "return", "turnover/yr"]).reindex(
        columns=pd.MultiIndex.from_product([["ret_med", "p_pos", "return", "turnover/yr"], list(SPLITS)]))
    print(piv.round(3).to_string())

    print("\n=== family medians across the grid (robustness: how the whole family behaves, not just its best) ===")
    med = df.groupby(["family", "split"])[["ret_med", "p_pos", "p_gt3", "maxdd_med", "return", "turnover/yr"]].median().round(3)
    print(med.unstack("split").reindex(columns=pd.MultiIndex.from_product(
        [["ret_med", "p_pos", "p_gt3", "maxdd_med", "return", "turnover/yr"], list(SPLITS)])).to_string())

    boots, texts = [], []
    for rank in RANKINGS:
        b, t = verdict(df, eqs, args.n_boot, families, rank)
        boots += b
        texts.append(t)
    pd.DataFrame(boots).to_csv(f_bs, index=False)
    print("\n".join(texts))
    (RESULTS / f"research_verdict{suf}.txt").write_text("\n".join(texts))
    print(f"\nwrote {f_csv}, {f_bs}, {f_eq}")


if __name__ == "__main__":
    main()
