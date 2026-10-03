"""Collect every number for the strategy report into results/report_data.json."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from bot.research.sharpe_opt import SPLITS, load_prices, metrics, split
from bot.strategies.risk_parity import RPParams, simulate, target_weights

FEE = 0.001


def win14(r, starts=None):
    eq = (1 + r).cumprod()
    starts = r.index[:-14] if starts is None else starts
    out = []
    for s in starts:
        seg = eq[s: s + pd.Timedelta(days=14)]
        if len(seg) < 14:
            continue
        x = seg.pct_change().dropna()
        tot = seg.iloc[-1] / seg.iloc[0] - 1
        sd = x.std(); dd = np.sqrt((np.minimum(x, 0) ** 2).mean()); mdd = (seg / seg.cummax() - 1).min()
        sh = x.mean() / sd * np.sqrt(365) if sd > 0 else 0.0
        so = x.mean() / dd * np.sqrt(365) if dd > 0 else 0.0
        ca = ((1 + tot) ** (365 / 14) - 1) / abs(mdd) if mdd < 0 else 0.0
        out.append((tot, mdd, 0.4 * so + 0.3 * sh + 0.3 * ca))
    a = np.array(out)
    return {"n": len(a), "ret_med": float(np.median(a[:, 0])), "p_pos": float((a[:, 0] > 0).mean()),
            "p_gt3": float((a[:, 0] > 0.03).mean()), "p_gt5": float((a[:, 0] > 0.05).mean()),
            "ret_p10": float(np.percentile(a[:, 0], 10)), "ret_p90": float(np.percentile(a[:, 0], 90)),
            "mdd_med": float(np.median(a[:, 1])), "mdd_p10": float(np.percentile(a[:, 1], 10)),
            "score_med": float(np.median(a[:, 2]))}


def main():
    P = load_prices().iloc[:-1]
    R = P.pct_change(fill_method=None).fillna(0.0)
    p = RPParams(sleeves=["BTC", "ETH", "GOLD"], budgets=[2, 2, 1], lookbacks=[200], halflife=30, target_vol=0.20)
    W = target_weights(P, p)
    books = {"Strategy (risk-parity trend, 20% vol)": simulate(P, W, 0.02, FEE)}
    books["BTC buy & hold"] = R["BTC"]
    books["ETH buy & hold"] = R["ETH"]
    books["50/50 BTC/ETH (rebal. at 5% drift)"] = simulate(P[["BTC", "ETH"]], pd.DataFrame(0.5, index=P.index, columns=["BTC", "ETH"]), 0.05, FEE)
    books["Gold (GLD/PAXG) hold"] = R["GOLD"]
    books["Nasdaq-100 (QQQ) hold"] = R["EQ"]
    books["Vol targeting only (no trend filter)"] = simulate(P, target_weights(P, p, trend=False), 0.02, FEE)
    lc = np.log(P[["BTC", "ETH"]])
    gate = (lc > lc.ewm(span=10, adjust=False).mean()).astype(float)
    Wg = gate / 2
    from bot.research.regime_study import vol_target
    Rc = P[["BTC", "ETH"]].pct_change(fill_method=None)
    Wg = vol_target(Wg, Rc, 0.3)
    books["Old: gated core, no short"] = simulate(P[["BTC", "ETH"]], Wg, 0.05, FEE)
    Ws = Wg.copy(); dn = lc["BTC"] < lc["BTC"].ewm(span=10, adjust=False).mean()
    Ws.loc[dn, ["BTC", "ETH"]] = 0.0; Ws.loc[dn, "BTC"] = -1.0
    Ws = vol_target(Ws, Rc, 0.3)
    books["Old: gated core + BTC short"] = simulate(P[["BTC", "ETH"]], Ws, 0.05, FEE)

    periods = dict(SPLITS); periods["full"] = ("2016-06-01", "2100-01-01")
    out = {"periods": {k: [v[0], min(v[1], str(P.index[-1].date()))] for k, v in periods.items()}, "metrics": {}, "w14": {}}
    for name, r in books.items():
        out["metrics"][name] = {}
        for sp, (s, e) in periods.items():
            x = r[(r.index >= s) & (r.index < e)]
            out["metrics"][name][sp] = {k: float(v) for k, v in metrics(x).items()}
        oos = r[r.index >= "2023-01-01"]
        out["w14"][name] = {"oos_2023_26": win14(oos), "full": win14(r[r.index >= "2016-06-01"])}
    # equity curves (weekly) and drawdowns from 2017
    eqs = pd.DataFrame({k: (1 + v[v.index >= "2017-01-01"]).cumprod() for k, v in books.items()})
    wk = eqs.resample("W").last()
    out["equity_weekly"] = {"dates": [d.strftime("%Y-%m-%d") for d in wk.index],
                            **{k: [round(float(x), 4) for x in wk[k]] for k in ["Strategy (risk-parity trend, 20% vol)", "BTC buy & hold", "Gold (GLD/PAXG) hold", "Vol targeting only (no trend filter)", "Old: gated core + BTC short"]}}
    dd = (eqs / eqs.cummax() - 1).resample("W").min()
    out["drawdown_weekly"] = {"dates": out["equity_weekly"]["dates"],
                              **{k: [round(float(x), 4) for x in dd[k]] for k in ["Strategy (risk-parity trend, 20% vol)", "BTC buy & hold"]}}
    # exposure history
    Wm = W[W.index >= "2017-01-01"].resample("W").mean()
    out["weights_weekly"] = {"dates": [d.strftime("%Y-%m-%d") for d in Wm.index],
                             **{c: [round(float(x), 4) for x in Wm[c]] for c in W.columns}}
    out["weights_today"] = {c: float(W.iloc[-1][c]) for c in W.columns}
    out["today"] = str(P.index[-1].date())
    # trading activity
    Wt = W[W.index >= "2023-01-01"]
    cur = np.zeros(3); trade_days = []
    Rv = P[W.columns].pct_change(fill_method=None).fillna(0).values[-len(Wt):]
    for i, (t, w) in enumerate(Wt.iterrows()):
        r = Rv[i]; pr = cur @ r
        if 1 + pr > 0:
            cur = cur * (1 + r) / (1 + pr)
        diff = w.values - cur
        tr = np.where(np.abs(diff) > 0.02, diff, 0.0); tr = np.where((w.values == 0) & (cur != 0), -cur, tr)
        n = int((tr != 0).sum()); trade_days.append(n)
        cur = cur + tr
    td = pd.Series(trade_days, index=Wt.index)
    win_trades = [int(td[s: s + pd.Timedelta(days=13)].sum()) for s in td.index[:-14]]
    win_days = [int((td[s: s + pd.Timedelta(days=13)] > 0).sum()) for s in td.index[:-14]]
    out["activity"] = {"orders_per_year": float(td.sum() / (len(td) / 365)),
                       "trade_days_per_year": float((td > 0).sum() / (len(td) / 365)),
                       "orders_per_14d_median": float(np.median(win_trades)),
                       "trade_days_per_14d_median": float(np.median(win_days)),
                       "share_14d_windows_with_no_trade": float(np.mean(np.array(win_trades) == 0)),
                       "share_14d_windows_with_trades_on_fewer_than_5_days": float(np.mean(np.array(win_days) < 5)),
                       "turnover_per_year": float(W[W.index >= "2023-01-01"].diff().abs().sum(1).sum() / (len(Wt) / 365)),
                       "avg_gross": float(W[W.index >= "2023-01-01"].sum(1).mean()),
                       "share_days_fully_in_cash": float((W[W.index >= "2017-01-01"].sum(1) == 0).mean())}
    # correlations (daily, 2017+)
    c = R[["BTC", "ETH", "GOLD", "EQ"]][R.index >= "2017-01-01"].corr()
    out["corr"] = {a: {b: float(c.loc[a, b]) for b in c.columns} for a in c.index}
    json.dump(out, open("results/report_data.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    for sp in ("test", "holdout", "full"):
        print(f"\n== {sp} ==")
        print(pd.DataFrame({k: v[sp] for k, v in out["metrics"].items()}).T[["CAGR", "vol", "Sharpe", "Sortino", "Calmar", "MaxDD", "composite"]].round(2).to_string())
    print("\n== 14d windows 2023-2026 ==")
    print(pd.DataFrame({k: v["oos_2023_26"] for k, v in out["w14"].items()}).T.round(3).to_string())
    print("\nactivity", {k: round(v, 3) for k, v in out["activity"].items()})
    print("weights today", out["weights_today"], "corr", c.round(2).to_dict())


if __name__ == "__main__":
    main()
