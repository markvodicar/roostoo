"""Regime study: which simple strategy families did best over 14-day windows, overall
and in market regimes resembling the one at competition start (Oct 2026)?

Daily bars (00:00 UTC closes from data/full hourly files). Weights decided at close t
earn the close t -> t+1 return (one-day lag). Fees on every unit of turnover.
Parameters are fixed, round numbers chosen a priori (no tuning) to limit overfitting.

    python3 -m bot.research.regime_study
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from bot.ml.windows import summarize

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data", "full")
FEE = 0.001          # taker, conservative (live executor mostly gets 0.05% maker)
START = "2019-01-01"
EXCLUDE = {"PAXG"}


def load_daily():
    cl, vol = {}, {}
    for f in glob.glob(os.path.join(DATA, "*.csv")):
        c = os.path.basename(f)[:-4]
        if c in EXCLUDE:
            continue
        d = pd.read_csv(f, index_col=0, parse_dates=True)
        cl[c] = d["close"].resample("1D").last()
        vol[c] = d["volume_usd"].resample("1D").sum(min_count=1)
    C = pd.DataFrame(cl).sort_index()
    V = pd.DataFrame(vol).reindex(C.index)
    # need 60 days of history before a coin is eligible
    seen = C.notna().cumsum()
    C = C.where(seen > 0)
    return C, V, seen


def liquid_universe(C, V, seen, n=20, min_hist=60):
    dv = V.rolling(30, min_periods=20).median()
    ok = (seen >= min_hist) & C.notna()
    rk = dv.where(ok).rank(axis=1, ascending=False)
    return rk <= n


def inv_vol(sel_mask, vol):
    iv = (1.0 / vol).where(sel_mask)
    return iv.div(iv.sum(1), axis=0).fillna(0.0)


def vol_target(w, R, target, lookback=30, cap=1.0):
    """Scale weights so ex-ante portfolio vol (sample cov over lookback) hits target."""
    out = w.copy()
    for i in range(lookback, len(w)):
        wi = w.iloc[i]
        names = wi[wi != 0].index
        if len(names) == 0:
            continue
        cov = R[names].iloc[i - lookback + 1: i + 1].cov().fillna(0).values * 365
        pv = float(np.sqrt(max(wi[names].values @ cov @ wi[names].values, 1e-12)))
        out.iloc[i] = wi * min(target / pv, cap / max(wi.abs().sum(), 1e-12))
    return out


def simulate(W, R, rebal=1, fee=FEE):
    """W: target weights at close t (DataFrame). Rebalance every `rebal` days."""
    W = W.reindex(R.index).fillna(0.0)
    w = pd.Series(0.0, index=R.columns)
    eq, curve = 1.0, []
    Rv = R.fillna(0.0).values
    for i in range(len(R)):
        r = Rv[i]
        pr = float(w.values @ r)
        eq *= 1 + pr
        if 1 + pr > 0:
            w = w * (1 + r) / (1 + pr)
        if i % rebal == 0:
            t = W.iloc[i]
            to = float((t - w).abs().sum())
            eq *= 1 - to * fee
            w = t.copy()
        curve.append(eq)
    return pd.Series(curve, index=R.index)


def strategies(C, V, seen):
    R = C.pct_change(fill_method=None)
    lc = np.log(C)
    vol = R.rolling(30, min_periods=20).std() * np.sqrt(365)
    ema10, ema20, ema50 = (lc.ewm(span=s, adjust=False).mean() for s in (10, 20, 50))
    liq = liquid_universe(C, V, seen, 20)
    liq10 = liquid_universe(C, V, seen, 10)
    S = {}
    z = pd.DataFrame(0.0, index=C.index, columns=C.columns)

    def one(c):
        w = z.copy(); w[c] = 1.0; return w
    S["btc_hold"] = (one("BTC"), 1)
    S["eth_hold"] = (one("ETH"), 1)
    maj = ["BTC", "ETH", "SOL", "BNB", "XRP"]
    w = z.copy(); m = C[maj].notna(); w[maj] = m.div(m.sum(1), axis=0); S["majors5_ew_hold"] = (w, 7)

    # gated BTC/ETH core (daily approximation of the launch strategy), vol target 0.3
    core = ["BTC", "ETH"]
    above = (lc[core] > ema10[core])
    w = z.copy(); w[core] = above.astype(float).div(2)
    S["core_gated"] = (vol_target(w, R, 0.3), 3)
    w2 = w.copy()
    btc_dn = lc["BTC"] < ema10["BTC"]
    w2.loc[btc_dn, core] = 0.0; w2.loc[btc_dn, "BTC"] = -1.0
    S["core_gated_short"] = (vol_target(w2, R, 0.3), 3)

    # broad time-series trend: every liquid coin above EMA20 and EMA50, inverse vol, vol target 0.5
    tr = liq & (lc > ema20) & (lc > ema50)
    S["trend_liquid20"] = (vol_target(inv_vol(tr, vol), R, 0.5).clip(upper=0.25), 1)
    S["trend_liquid20_v1"] = (vol_target(inv_vol(tr, vol), R, 1.0).clip(upper=0.25), 1)

    # cross-sectional momentum: top 5 liquid coins by 30d return, must be above EMA20,
    # only when breadth (share of liquid coins above EMA50) > 0.5
    r30 = C / C.shift(30) - 1
    breadth = ((lc > ema50) & liq).sum(1) / liq.sum(1).replace(0, np.nan)
    cand = r30.where(liq & (lc > ema20))
    top5 = cand.rank(axis=1, ascending=False) <= 5
    on = breadth > 0.5
    w = inv_vol(top5, vol).mul(on, axis=0)
    S["xsmom_top5"] = (vol_target(w, R, 1.0).clip(upper=0.35), 3)
    r7 = C / C.shift(7) - 1
    top5b = r7.where(liq & (lc > ema20)).rank(axis=1, ascending=False) <= 5
    S["xsmom7_top5"] = (vol_target(inv_vol(top5b, vol).mul(on, axis=0), R, 1.0).clip(upper=0.35), 3)

    # equal-weight liquid top10 basket when BTC above EMA20
    btc_up = lc["BTC"] > ema20["BTC"]
    w = liq10.astype(float).div(liq10.sum(1), axis=0).mul(btc_up, axis=0).fillna(0)
    S["ew_liquid10_btcgate"] = (w, 3)

    # breakout: enter at 20d closing high, exit below 10d low; inverse vol, vol target 0.5
    hi20 = C.rolling(20).max(); lo10 = C.rolling(10).min()
    state = pd.DataFrame(False, index=C.index, columns=C.columns)
    cur = pd.Series(False, index=C.columns)
    for i in range(len(C)):
        cur = (cur | (C.iloc[i] >= hi20.iloc[i])) & ~(C.iloc[i] <= lo10.iloc[i]) & liq.iloc[i]
        state.iloc[i] = cur
    S["breakout_liquid20"] = (vol_target(inv_vol(state, vol), R, 0.5).clip(upper=0.25), 1)
    return S, R, dict(breadth=breadth, lc=lc, ema50=ema50, vol=vol, liq=liq)


def regime_flags(C, aux):
    lc = aux["lc"]
    btc = C["BTC"]
    f = pd.DataFrame(index=C.index)
    f["btc_gt_200d"] = btc > btc.rolling(200).mean()
    f["btc_90d"] = btc / btc.shift(90) - 1
    f["btc_from_365d_high"] = btc / btc.rolling(365).max() - 1
    f["breadth50"] = aux["breadth"]
    f["btc_vol30"] = np.log(btc).diff().rolling(30).std() * np.sqrt(365)
    # "now-like": recovery rally inside a longer drawdown, broad participation
    f["now_like"] = (f.btc_gt_200d & (f.btc_90d > 0.15) & (f.btc_from_365d_high < -0.15)
                     & (f.breadth50 > 0.6))
    return f


def window_stats(eq, starts):
    rows = []
    for s in starts:
        seg = eq[s: s + pd.Timedelta(days=14)]
        if len(seg) < 14:
            continue
        r = seg.pct_change().dropna()
        sh = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else 0.0
        dd = np.sqrt((np.minimum(r, 0) ** 2).mean())
        so = r.mean() / dd * np.sqrt(365) if dd > 0 else 0.0
        tot = seg.iloc[-1] / seg.iloc[0] - 1
        mdd = float((seg / seg.cummax() - 1).min())
        ann = (1 + tot) ** (365 / 14) - 1
        ca = ann / abs(mdd) if mdd < 0 else 0.0
        rows.append({"ret": tot, "score": 0.4 * so + 0.3 * sh + 0.3 * ca, "mdd": mdd})
    return pd.DataFrame(rows)


def describe(ws):
    if ws.empty:
        return {}
    r = ws.ret
    return {"n": len(ws), "ret_mean": r.mean(), "ret_med": r.median(), "p_pos": (r > 0).mean(),
            "p_gt3": (r > 0.03).mean(), "p_gt5": (r > 0.05).mean(), "ret_p10": r.quantile(0.1),
            "mdd_med": ws.mdd.median(), "score_med_pos": ws.score[r > 0].median() if (r > 0).any() else np.nan}


def main():
    C, V, seen = load_daily()
    S, R, aux = strategies(C, V, seen)
    flags = regime_flags(C, aux)
    now = flags.iloc[-1]
    print("Regime today:", now.round(3).to_dict())
    idx = C.index[(C.index >= START) & (C.index <= C.index[-1] - pd.Timedelta(days=14))]
    sets = {
        "all_2019_26": idx,
        "now_like": idx[flags.now_like.reindex(idx).fillna(False).values],
        "last_365d": idx[idx >= C.index[-1] - pd.Timedelta(days=365)],
        "oct_starts": idx[idx.month == 10],
    }
    print({k: len(v) for k, v in sets.items()})
    print("now_like episodes (month starts):",
          sorted(set(d.strftime("%Y-%m") for d in sets["now_like"])))
    rows = []
    for name, (W, rebal) in S.items():
        eq = simulate(W.loc[:, R.columns], R, rebal)
        for sname, st in sets.items():
            rows.append({"strategy": name, "set": sname, **describe(window_stats(eq, st))})
        eq.to_frame("eq").to_parquet(f"results/regime_eq_{name}.parquet")
    df = pd.DataFrame(rows)
    df.to_csv("results/regime_study.csv", index=False)
    pd.set_option("display.width", 250)
    for sname in sets:
        print(f"\n=== {sname} ===")
        print(df[df.set == sname].drop(columns="set").set_index("strategy").round(3).to_string())


if __name__ == "__main__":
    main()
