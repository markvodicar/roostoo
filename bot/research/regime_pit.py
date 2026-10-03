"""Survivorship-free re-test of the regime study and the proposed regime blend.

Universe: every USDT spot pair Binance ever listed (bot/research/pit_data.py), including
delisted and collapsed coins. Each day the tradeable "liquid-20" is the top 20 by
trailing-30d median dollar volume among coins with >= 60 days of history, chosen with
data up to that day only. "Majors" = top 5 by trailing-180d median dollar volume
(point-in-time), not today's hindsight list. A coin whose series ends is sold at its last
close (the simulator drops it at the next rebalance with zero further return).

Inference uses DAILY strategy returns (weights decided at close t earn t -> t+1), with
Newey-West (HAC) t-statistics and a stationary block bootstrap; 14-day window statistics
are reported descriptively.

    python3 -m bot.research.regime_pit
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot.research.pit_data import load
from bot.research.regime_study import describe, inv_vol, simulate, vol_target, window_stats

START = "2018-06-01"
FEE = 0.001
RNG = np.random.default_rng(0)


def panel():
    a = load()
    C = pd.DataFrame({k: v["close"] for k, v in a.items()}).sort_index()
    V = pd.DataFrame({k: v["qv"] for k, v in a.items()}).reindex(C.index)
    C.index = C.index.tz_convert(None) if C.index.tz is not None else C.index
    V.index = C.index
    # drop single-day absurd prints (>20x or <1/20 vs both neighbours): bad ticks in archives
    r = C / C.shift(1)
    bad = ((r > 20) & (C.shift(-1) / C < 0.1)) | ((r < 0.05) & (C.shift(-1) / C > 10))
    C = C.mask(bad)
    return C, V


def rank_universe(C, V, n, win, min_hist=60):
    seen = C.notna().cumsum()
    dv = V.rolling(win, min_periods=int(win * 0.6)).median()
    ok = (seen >= min_hist) & C.notna()
    return dv.where(ok).rank(axis=1, ascending=False, method="first") <= n


def build(C, V):
    R = C.pct_change(fill_method=None).clip(-0.95, 5.0)
    lc = np.log(C)
    vol = R.rolling(30, min_periods=20).std() * np.sqrt(365)
    ema10, ema20, ema50 = (lc.ewm(span=s, adjust=False).mean() for s in (10, 20, 50))
    liq = rank_universe(C, V, 20, 30)
    maj = rank_universe(C, V, 5, 180)
    z = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    B = {}
    B["btc_hold"] = z.copy().assign(BTC=1.0)
    B["majors_pit_ew"] = maj.astype(float).div(maj.sum(1).replace(0, np.nan), axis=0).fillna(0)
    B["liquid20_ew"] = liq.astype(float).div(liq.sum(1).replace(0, np.nan), axis=0).fillna(0)
    core = ["BTC", "ETH"]
    w = z.copy(); w[core] = (lc[core] > ema10[core]).astype(float) / 2
    B["core_gated"] = vol_target(w, R, 0.3)
    w2 = w.copy(); dn = lc["BTC"] < ema10["BTC"]
    w2.loc[dn, core] = 0.0; w2.loc[dn, "BTC"] = -1.0
    B["launch_core_gated_short"] = vol_target(w2, R, 0.3)
    tr = liq & (lc > ema20) & (lc > ema50)
    B["trend_liquid20"] = vol_target(inv_vol(tr, vol), R, 0.5).clip(upper=0.25)
    B["trend_liquid20_v1"] = vol_target(inv_vol(tr, vol), R, 1.0).clip(upper=0.25)
    breadth = ((lc > ema50) & liq).sum(1) / liq.sum(1).replace(0, np.nan)
    r30 = C / C.shift(30) - 1
    top5 = r30.where(liq & (lc > ema20)).rank(axis=1, ascending=False) <= 5
    B["xsmom_top5"] = vol_target(inv_vol(top5, vol).mul(breadth > 0.5, axis=0), R, 1.0).clip(upper=0.35)
    btc = C["BTC"]
    bull = ((btc > btc.rolling(200).mean()) & (breadth > 0.5)).astype(float)
    bear = ((btc <= btc.rolling(200).mean()) & (breadth < 0.3)).astype(float)
    mixed = 1 - bull - bear
    B["PROPOSAL_regime_blend"] = ((B["majors_pit_ew"] * 0.5 + B["trend_liquid20_v1"] * 0.25
                                   + B["xsmom_top5"] * 0.25).mul(bull, axis=0)
                                  + B["trend_liquid20"].mul(mixed, axis=0)
                                  + B["launch_core_gated_short"].mul(bear, axis=0))
    # placebo: same blend, held in every regime (isolates the value of regime switching)
    B["blend_always_on"] = (B["majors_pit_ew"] * 0.5 + B["trend_liquid20_v1"] * 0.25
                            + B["xsmom_top5"] * 0.25)
    flags = pd.DataFrame({"bull": bull, "bear": bear, "mixed": mixed, "breadth": breadth,
                          "btc_90d": btc / btc.shift(90) - 1,
                          "btc_from_high": btc / btc.rolling(365, min_periods=200).max() - 1})
    flags["now_like"] = ((bull > 0) & (flags.btc_90d > 0.15) & (flags.btc_from_high < -0.15)
                         & (breadth > 0.6)).astype(float)
    return B, R, flags, liq, maj


def nw_t(x, lags=10):
    """Newey-West t-statistic of the mean of x."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x); m = x.mean(); e = x - m
    s = e @ e / n
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return m / np.sqrt(s / n), n


def block_boot_p(x, block=20, n_boot=4000):
    """One-sided stationary-bootstrap p-value for mean(x) > 0 (centred under the null)."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x); m = x.mean(); xc = x - m
    p = 1.0 / block
    cnt = 0
    for _ in range(n_boot):
        idx = np.empty(n, int); i = RNG.integers(n)
        for k in range(n):
            idx[k] = i
            i = RNG.integers(n) if RNG.random() < p else (i + 1) % n
        cnt += xc[idx].mean() >= m
    return (cnt + 1) / (n_boot + 1)


def episodes(mask):
    """Number of separate runs of True (independent regime episodes)."""
    m = np.asarray(mask, bool)
    return int(((m[1:] & ~m[:-1]).sum()) + (1 if len(m) and m[0] else 0))


def main():
    C, V = panel()
    print(f"assets {C.shape[1]} (incl. delisted), days {C.shape[0]}, {C.index[0]:%Y-%m-%d} -> {C.index[-1]:%Y-%m-%d}")
    B, R, flags, liq, maj = build(C, V)
    first = C.index >= START
    alive_end = C.iloc[-1].notna()
    ever_liq = liq[first].any()
    print(f"coins ever in point-in-time liquid-20 since {START}: {int(ever_liq.sum())}, "
          f"of which no longer trading today: {int((ever_liq & ~alive_end).sum())}")
    print("majors (top-5 by 180d volume) on sample dates:",
          {d: sorted(c for c in maj.columns[maj.loc[d]])
           for d in ["2018-06-01", "2020-01-01", "2021-06-01", "2023-01-01", "2026-10-02"]})
    print("today regime:", flags.iloc[-1].round(3).to_dict())
    eqs, daily = {}, {}
    for k, W in B.items():
        eq = simulate(W.fillna(0.0), R, 1, FEE)
        eq = eq[first]
        eqs[k] = eq / eq.iloc[0]
        daily[k] = eqs[k].pct_change()
    D = pd.DataFrame(daily).iloc[1:]
    reg = flags.shift(1).reindex(D.index)          # regime known at the previous close
    pd.DataFrame(eqs).to_parquet("results/pit_equity.parquet")

    # ---- descriptive: full-period and 14-day windows ----
    rows = []
    idx = D.index[D.index <= D.index[-1] - pd.Timedelta(days=14)]
    sets = {"all": idx, "bull": idx[flags.bull.reindex(idx).values > 0],
            "now_like": idx[flags.now_like.reindex(idx).values > 0]}
    for k, eq in eqs.items():
        r = eq.pct_change().dropna(); yrs = (eq.index[-1] - eq.index[0]).days / 365
        cagr = eq.iloc[-1] ** (1 / yrs) - 1; mdd = (eq / eq.cummax() - 1).min()
        sh = r.mean() / r.std() * np.sqrt(365); so = r.mean() / np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(365)
        row = {"strategy": k, "CAGR": cagr, "Sharpe": sh, "Sortino": so, "Calmar": cagr / abs(mdd), "MaxDD": mdd}
        for sn, st in sets.items():
            d = describe(window_stats(eq, st))
            row.update({f"{sn}_med14": d["ret_med"], f"{sn}_ppos": d["p_pos"], f"{sn}_p10": d["ret_p10"]})
        rows.append(row)
    desc = pd.DataFrame(rows).set_index("strategy")
    pd.set_option("display.width", 260)
    print("\n=== Survivorship-free backtest", START, "->", D.index[-1].date(), "(daily, 0.10% fee) ===")
    print(desc.round(3).to_string())
    print({k: f"{len(v)} windows / {episodes(flags[k].reindex(D.index).values > 0) if k != 'all' else 1} episodes"
           for k, v in sets.items()})

    # ---- inference on daily returns ----
    tests = []
    comps = {"PROPOSAL_regime_blend": ["launch_core_gated_short", "btc_hold", "majors_pit_ew", "blend_always_on"]}
    for cond in ("all", "bull", "now_like"):
        m = np.ones(len(D), bool) if cond == "all" else (reg[cond].values > 0)
        n_ep = 1 if cond == "all" else episodes(m)
        for k in ["PROPOSAL_regime_blend", "launch_core_gated_short", "btc_hold", "majors_pit_ew", "trend_liquid20",
                  "xsmom_top5"]:
            x = D[k].values[m]
            t, n = nw_t(x)
            tests.append({"cond": cond, "test": f"{k} mean > 0", "days": n, "episodes": n_ep,
                          "ann_mean": np.nanmean(x) * 365, "ann_sharpe": np.nanmean(x) / np.nanstd(x) * np.sqrt(365),
                          "NW_t": t, "boot_p": block_boot_p(x)})
        for a, bs in comps.items():
            for b in bs:
                x = (D[a] - D[b]).values[m]
                t, n = nw_t(x)
                tests.append({"cond": cond, "test": f"{a} - {b} > 0", "days": n, "episodes": n_ep,
                              "ann_mean": np.nanmean(x) * 365, "ann_sharpe": np.nanmean(x) / np.nanstd(x) * np.sqrt(365),
                              "NW_t": t, "boot_p": block_boot_p(x)})
    # regime timing: does the long book earn more in bull regimes than otherwise?
    for k in ["majors_pit_ew", "blend_always_on", "btc_hold"]:
        bm = reg["bull"].values > 0
        x, y = D[k].values[bm], D[k].values[~bm]
        diff = np.nanmean(x) - np.nanmean(y)
        se = np.sqrt((nw_t(x)[0] and (np.nanmean(x) / nw_t(x)[0]) ** 2) + (np.nanmean(y) / nw_t(y)[0]) ** 2)
        tests.append({"cond": "bull vs not", "test": f"{k}: mean(bull) - mean(not bull)", "days": int(bm.sum()),
                      "episodes": episodes(bm), "ann_mean": diff * 365, "ann_sharpe": np.nan,
                      "NW_t": diff / se, "boot_p": np.nan})
    T = pd.DataFrame(tests)
    T.to_csv("results/pit_tests.csv", index=False)
    desc.to_csv("results/pit_desc.csv")
    n_tried = 25
    print(f"\n=== Significance (daily returns; one-sided; Bonferroni bar for ~{n_tried} strategies tried: p < {0.05 / n_tried:.4f}, |t| > 3.1) ===")
    print(T.round(4).to_string(index=False))

    # ---- non-overlapping 14d windows in bull regime, averaged over the 14 phase offsets ----
    print("\n=== Non-overlapping 14-day windows starting in bull regime (14 phase offsets) ===")
    for k in ["PROPOSAL_regime_blend", "launch_core_gated_short", "btc_hold", "majors_pit_ew"]:
        res = []
        for off in range(14):
            st = sets["bull"][off::14]
            w = window_stats(eqs[k], st).ret
            res.append((len(w), w.mean(), w.std(ddof=1) / np.sqrt(len(w)), (w > 0).mean()))
        n, mu, se, pp = np.array(res).mean(0)
        print(f"  {k:28s} windows/offset {n:.0f}  mean 14d {mu:+.3%}  t {mu / se:+.2f}  p_pos {pp:.2f}")


if __name__ == "__main__":
    main()
