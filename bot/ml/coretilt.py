"""Core + tilt portfolio: a vol-targeted, regime-gated core in the majors plus a
CNN-selected tilt. Motivation (from the test diagnostics, so test-informed):
the CNN ranks alts better than chance (+0.085%/day vs the alt universe) but the
alt universe trails BTC, so a pure alt-picking book cannot beat BTC in a bull
market. The core captures market direction; the tilt adds the ranking edge.

    python -m bot.ml.coretilt --tag ens
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from bot.backtest import metrics
from bot.ml.dataset import get_panel
from bot.ml.evaluate import frames_for, predictions
from bot.ml.features import HORIZONS
from bot.ml.portfolio import HPY, PortParams, report, simulate, weights_from_scores

CORE = ["BTC", "ETH"]


def make_weight_fn(lcf: pd.DataFrame, core_frac: float, ema_span: int, core_vol: float):
    """Returns weight_fn(scores, vol, p, i). Core = BTC/ETH equal-weight, gated by
    price > EMA(ema_span) per coin (computed on lcf up to row i), vol-targeted to core_vol.
    Tilt = weights_from_scores on the alt universe with budget (1 - core_frac)."""
    ema = lcf[CORE].ewm(span=ema_span, adjust=False).mean()
    above = (lcf[CORE] > ema).astype(float).values
    core_cols = [lcf.columns.get_loc(c) for c in CORE]

    def wf(scores: pd.Series, vol: pd.Series, p: PortParams, i: int) -> pd.Series:
        w = pd.Series(0.0, index=scores.index)
        on = above[i]
        if on.sum() > 0:
            cv = vol[CORE].values * np.sqrt(HPY)
            cw = on / on.sum()
            pv = float(np.sqrt(((cw * cv) ** 2).sum() + 0.8 * ((cw * cv).sum() ** 2 - ((cw * cv) ** 2).sum())))
            w[CORE] = cw * min(core_vol / max(pv, 1e-9), 1.0) * core_frac
        if core_frac < 1.0:
            tilt = weights_from_scores(scores.drop(CORE, errors="ignore"), vol, p) * (1 - core_frac)
            w = w.add(tilt, fill_value=0.0)
        g = w.abs().sum()
        if g > 1.0:
            w /= g
        return w

    return wf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ens")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    feat, vol, lc, valid, times, coins = get_panel()
    rows = []
    for split in ("val", "test", "holdout"):
        preds = predictions(args.tag, split, feat, valid, times, coins)
        volf, lcf, mkt = frames_for(split, preds[HORIZONS[0]], vol, lc, feat, times, coins)
        lcf = lcf.ffill()
        combos = itertools.product((1.0, 0.7), (24 * 10, 24 * 30), (0.3, 0.5), (24, 336), (3, 5), (48, 72), (0.10,))
        for core_frac, ema_span, core_vol, head, top_k, rebal, band in combos:
            if core_frac == 1.0 and (head != 24 or top_k != 3):
                continue  # pure core: tilt params irrelevant
            p = PortParams(top_k=top_k, head=head, rebal_h=rebal, band=band, target_vol=0.5, thresh=0.0)
            wf = make_weight_fn(lcf, core_frac, ema_span, core_vol)
            eq, tr = simulate(preds[head], volf, lcf, p, weight_fn=wf)
            name = f"core{core_frac:.1f} ema{ema_span // 24}d cv{core_vol} rb{rebal} b{band} | tilt head{head} k{top_k}"
            rows.append({"split": split, **report(name, eq), "trades": len(tr),
                         "turnover/yr": round(tr.turnover.sum() / max((eq.index[-1] - eq.index[0]).days / 365, 0.1), 1)})
        btc = np.exp(lcf["BTC"]).dropna()
        rows.append({"split": split, **report("btc_hold", btc / btc.iloc[0]), "trades": 0, "turnover/yr": 0})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    cols = ["return", "max_dd", "score", "win14_med_ret", "win14_med_score", "win14_pos", "turnover/yr"]
    piv = df.pivot(index="name", columns="split", values=cols).reindex(
        columns=pd.MultiIndex.from_product([cols, ["val", "test", "holdout"]]))
    print(piv.round(3).to_string())
    df.to_csv(f"results/coretilt_{args.tag}{('_' + args.out) if args.out else ''}.csv", index=False)


if __name__ == "__main__":
    main()
