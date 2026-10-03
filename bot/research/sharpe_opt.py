"""Sharpe-first long-only portfolio research (no shorts, no leverage).

Sleeves (all tradeable on Roostoo):
  BTC, ETH            crypto (data/full hourly -> 00:00 UTC daily closes, 2016+)
  GOLD                GLD price until 2020-08, PAXG after (live: PAXG/USD)
  EQ                  backtest: QQQ price (unbiased index); live: basket of large-cap stock
                      tokens replicating it. Weekends: price carried (zero return).
Construction, decided at close t, earning t -> t+1:
  trend_i  = mean over lookbacks L of 1[price_i > SMA_L(price_i)]          in [0, 1]
  vol_i    = EWMA daily vol (halflife h), annualised with sqrt(365)
  raw_i    = budget_i * trend_i / vol_i                                    (risk parity x trend)
  scale    = min(target_vol / ex-ante portfolio vol (EWMA cov), 1 / sum raw)  (no leverage)
  trade only when |target - current| > band (per sleeve)
Fees 0.10% per unit turnover (conservative; live mostly gets 0.05% maker).

Split (pre-registered): train 2016-2021, val 2022, test 2023-2025, holdout 2026-01..now.
Selection: maximise mean(composite_train, composite_val), composite = 0.4 Sortino + 0.3
Sharpe + 0.3 Calmar on daily returns. Test and holdout are reported, never used to choose.

    python3 -m bot.research.sharpe_opt
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd

from bot.strategies.risk_parity import RPParams, simulate, target_weights

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
FEE = 0.001
SPLITS = {"train": ("2016-06-01", "2022-01-01"), "val": ("2022-01-01", "2023-01-01"),
          "test": ("2023-01-01", "2026-01-01"), "holdout": ("2026-01-01", "2100-01-01")}


def load_prices():
    def crypto(c):
        d = pd.read_csv(os.path.join(ROOT, "data", "full", f"{c}.csv"), index_col=0, parse_dates=True)
        s = d["close"].resample("1D").last()
        s.index = s.index.tz_convert(None)
        return s

    def stock(t):
        d = pd.read_csv(os.path.join(ROOT, "data", "stocks", f"{t}.csv"), index_col=0, parse_dates=True)
        return d["close"]
    P = pd.DataFrame({"BTC": crypto("BTC"), "ETH": crypto("ETH")})
    idx = pd.date_range("2016-01-01", P.index[-1], freq="D")
    P = P.reindex(idx)
    # gold: GLD before PAXG existed, PAXG (the asset traded live, 24/7) from 2020-09
    gld = stock("GLD").reindex(idx).ffill()
    paxg = crypto("PAXG").reindex(idx)
    cut = pd.Timestamp("2020-09-01")
    link = paxg.loc[cut] / gld.loc[cut]
    P["GOLD"] = pd.concat([gld[gld.index < cut] * link, paxg[paxg.index >= cut]])
    P["EQ"] = stock("QQQ").reindex(idx).ffill()
    return P


def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 20:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / 365
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    sd = r.std()
    dsd = np.sqrt((np.minimum(r, 0) ** 2).mean())
    mdd = float((eq / eq.cummax() - 1).min())
    sh = r.mean() / sd * np.sqrt(365) if sd > 0 else 0.0
    so = r.mean() / dsd * np.sqrt(365) if dsd > 0 else 0.0
    ca = cagr / abs(mdd) if mdd < 0 else 0.0
    return {"CAGR": cagr, "vol": sd * np.sqrt(365), "Sharpe": sh, "Sortino": so, "Calmar": ca,
            "MaxDD": mdd, "composite": 0.4 * so + 0.3 * sh + 0.3 * ca}


def run(P, sleeves, budgets, lookbacks, halflife, target_vol, band=0.02, fee=FEE, trend=True):
    """Backtest via the shared live weight function (bot.strategies.risk_parity)."""
    p = RPParams(sleeves=list(sleeves), budgets=list(budgets), lookbacks=list(lookbacks),
                 halflife=halflife, target_vol=target_vol, band=band).validate()
    W = target_weights(P, p, trend=trend)
    return simulate(P, W, band, fee), W


def split(r, name):
    s, e = SPLITS[name]
    return r[(r.index >= s) & (r.index < e)]


def grid():
    sleeve_sets = {"crypto": ["BTC", "ETH"], "crypto+gold": ["BTC", "ETH", "GOLD"],
                   "crypto+eq": ["BTC", "ETH", "EQ"], "crypto+eq+gold": ["BTC", "ETH", "EQ", "GOLD"]}
    budgets = {"equal_risk": None, "crypto_heavy": 2.0}   # crypto sleeves get 2x risk budget
    lookbacks = {"slow200": (200,), "multi": (20, 60, 120, 250), "fast": (10, 20, 50)}
    for (sn, sl), (bn, bm), (ln, lb), hl, tv in itertools.product(
            sleeve_sets.items(), budgets.items(), lookbacks.items(), (10, 30), (0.10, 0.15, 0.20, 0.30)):
        if sn == "crypto" and bn != "equal_risk":
            continue
        b = [(bm if (bm and s in ("BTC", "ETH")) else 1.0) for s in sl]
        yield {"sleeves": sn, "budget": bn, "lookbacks": ln, "halflife": hl, "target_vol": tv}, (sl, b, lb, hl, tv)


def main():
    P = load_prices()
    rows, rets = [], {}
    for meta, (sl, b, lb, hl, tv) in grid():
        r, _ = run(P, sl, b, lb, hl, tv)
        key = "|".join(f"{k}={v}" for k, v in meta.items())
        rets[key] = r
        row = dict(meta)
        for sp in SPLITS:
            m = metrics(split(r, sp))
            row.update({f"{sp}_{k}": v for k, v in m.items()})
        rows.append(row)
    # benchmarks
    from bot.research.regime_study import simulate  # noqa: F401  (keeps import graph explicit)
    bench = {}
    R = P.pct_change(fill_method=None).fillna(0)
    bench["BTC hold"] = R["BTC"]
    bench["QQQ hold"] = R["EQ"]
    bench["GLD hold"] = R["GOLD"]
    bench["static risk parity (no trend), crypto+eq+gold, tv0.15"] = run(
        P, ["BTC", "ETH", "EQ", "GOLD"], [1, 1, 1, 1], (200,), 30, 0.15, trend=False)[0]
    # launch strategy without the short (gated BTC/ETH, EMA10d, vol 0.3) ~ daily approximation
    lc = np.log(P[["BTC", "ETH"]])
    gate = (lc > lc.ewm(span=10, adjust=False).mean()).astype(float)
    bench["gated BTC/ETH core, no short (vol 0.3)"] = run(P.assign(), ["BTC", "ETH"], [1, 1], (10,), 30, 0.3)[0]
    for k, r in bench.items():
        row = {"sleeves": k, "budget": "-", "lookbacks": "-", "halflife": "-", "target_vol": "-"}
        for sp in SPLITS:
            row.update({f"{sp}_{kk}": v for kk, v in metrics(split(r, sp)).items()})
        rows.append(row)
        rets["BENCH " + k] = r
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", "sharpe_grid.csv"), index=False)
    pd.DataFrame(rets).to_parquet(os.path.join(ROOT, "results", "sharpe_grid_returns.parquet"))
    cfg = df[df.budget != "-"].copy()
    cfg["select"] = (cfg.train_composite + cfg.val_composite) / 2
    cfg = cfg.sort_values("select", ascending=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
    show = ["sleeves", "budget", "lookbacks", "halflife", "target_vol", "select"] + \
           [f"{sp}_{m}" for sp in ("train", "val", "test", "holdout") for m in ("CAGR", "Sharpe", "MaxDD")]
    print("Top 12 configs by pre-registered selection (train+val composite):")
    print(cfg[show].head(12).round(2).to_string(index=False))
    print("\nMean over ALL configs by sleeve set (robustness, no selection):")
    print(cfg.groupby("sleeves")[[f"{sp}_Sharpe" for sp in SPLITS]].mean().round(2).to_string())
    print(cfg.groupby("lookbacks")[[f"{sp}_Sharpe" for sp in SPLITS]].mean().round(2).to_string())
    print("\nBenchmarks:")
    print(df[df.budget == "-"][["sleeves"] + [f"{sp}_{m}" for sp in SPLITS for m in ("CAGR", "Sharpe", "MaxDD")]]
          .round(2).to_string(index=False))


if __name__ == "__main__":
    main()
