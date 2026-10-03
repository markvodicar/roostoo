"""Final bias audit of the deployed always-active configuration (config/strategy.json).

    python3 -m bot.research.final_audit      -> results/final_audit.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import bot.replay as rp
from bot.research.sharpe_opt import SPLITS, load_prices, metrics
from bot.strategies.risk_parity import RPParams, simulate, target_weights

ROOT = Path(__file__).resolve().parents[2]
RNG = np.random.default_rng(11)


def cfg_params(**over):
    d = json.load(open(ROOT / "config" / "strategy.json"))["risk_parity"]
    d = {**d, "sleeves": ["BTC", "ETH", "GOLD"], **over}
    return RPParams(**d)


def split(r, k):
    s, e = SPLITS[k]
    return r[(r.index >= s) & (r.index < e)]


def replay_metrics(eq):
    e = pd.concat([pd.Series([1e5]), eq.reset_index(drop=True)]); r = e.pct_change().dropna()
    tot = e.iloc[-1] / 1e5 - 1; ann = (1 + tot) ** (365 / len(r)) - 1; mdd = (e / e.cummax() - 1).min()
    dsd = np.sqrt((np.minimum(r, 0) ** 2).mean())
    return {"total": tot, "CAGR": ann, "Sharpe": r.mean() / r.std() * np.sqrt(365),
            "Sortino": r.mean() / dsd * np.sqrt(365), "Calmar": ann / abs(mdd), "MaxDD": mdd}


def main():
    out = {}
    P = load_prices().iloc[:-1]
    p = cfg_params()
    W = target_weights(P, p)
    sim = lambda W_, fee=0.001: simulate(P, W_, p.band, fee, force_daily=True, min_trade=p.min_trade_frac)  # noqa: E731
    base = sim(W)
    R = P.pct_change(fill_method=None).fillna(0)
    out["daily_backtest"] = {k: {"strategy": metrics(split(base, k)), "btc": metrics(split(R["BTC"], k))} for k in SPLITS}
    full = base[base.index >= "2016-06-01"]
    out["daily_backtest"]["full"] = {"strategy": metrics(full), "btc": metrics(R["BTC"][R.index >= "2016-06-01"])}

    # 1 causality
    ok = 0
    for T in RNG.choice(np.arange(400, len(P)), 25, replace=False):
        ok += np.allclose(target_weights(P.iloc[: T + 1], p).iloc[-1].values, W.iloc[T].values, atol=1e-6)
    out["causality_ok"] = int(ok)
    # 2 extra lag, 3 fees
    out["lag1"] = {k: metrics(split(sim(W.shift(1).fillna(0)), k))["Sharpe"] for k in ("test", "holdout")}
    out["fees"] = {f: {k: metrics(split(sim(W, f), k))["Sharpe"] for k in ("test", "holdout")} for f in (0.002, 0.003)}
    # 4 placebo: same machinery (floor, cap, vol target), random persistent trend states
    on = (W > W.min()).astype(int)
    sw = (target_weights(P, cfg_params(trend_floor=0.0)).gt(0).astype(int).diff().abs().sum() / len(P)).clip(lower=1 / 365)
    st = metrics(split(base, "test"))["Sharpe"]; pl = []
    for _ in range(300):
        sig = pd.DataFrame({c: (pd.Series(RNG.random(len(P)) < sw[c]).cumsum() % 2).values for c in p.sleeves},
                           index=P.index).astype(float)
        pl.append(metrics(split(sim(target_weights(P, p, signal=sig)), "test"))["Sharpe"])
    pl = np.array(pl)
    out["placebo"] = {"median": float(np.median(pl)), "p95": float(np.percentile(pl, 95)), "strategy": st,
                      "p": float(np.mean(pl >= st))}

    # 5 replay of the live code, test period, with execution-model stress
    cfg = str(ROOT / "config" / "strategy.json")
    runs = {}
    runs["baseline"] = rp.run("2023-01-01", "2026-01-01", cfg, seed=0)
    old_p, old_hs, old_price = rp.P_FILL_180S, dict(rp.HALF_SPREAD), rp.FakeRoostoo.price
    rp.P_FILL_180S = 0.5
    runs["limit fill 50% (vs 89%)"] = rp.run("2023-01-01", "2026-01-01", cfg, seed=0)
    rp.P_FILL_180S = old_p
    rp.HALF_SPREAD.update({k: 5e-4 for k in rp.HALF_SPREAD})
    runs["spread 10 bp (vs ~1-2 bp)"] = rp.run("2023-01-01", "2026-01-01", cfg, seed=0)
    rp.HALF_SPREAD.update(old_hs)

    def price_open(self, pair):
        t = pd.Timestamp(self.clock(), unit="s").floor("h")
        return float(self.h.at[t, (pair, "open")])
    rp.FakeRoostoo.price = price_open
    runs["price = hour open (no intra-hour path)"] = rp.run("2023-01-01", "2026-01-01", cfg, seed=0)
    rp.P_FILL_180S = 0.5; rp.HALF_SPREAD.update({k: 5e-4 for k in rp.HALF_SPREAD})
    runs["all three combined"] = rp.run("2023-01-01", "2026-01-01", cfg, seed=0)
    rp.FakeRoostoo.price = old_price; rp.P_FILL_180S = old_p; rp.HALF_SPREAD.update(old_hs)
    out["replay_stress"] = {k: {**replay_metrics(v["equity"]), "days_without_fill": len(v["days_without_fills"]),
                                "max_gap_h": v["max_gap_h"], "fees": v["fees"],
                                "maker_share": v["fills"]["MAKER"] / sum(v["fills"].values())} for k, v in runs.items()}
    rb = runs["baseline"]["equity"].pct_change().dropna()
    out["replay_vs_daily_corr"] = float(rb.corr(base.reindex(rb.index)))
    hol = rp.run("2026-01-01", "2026-10-03", cfg, seed=0)
    out["replay_holdout"] = {**replay_metrics(hol["equity"]), "days_without_fill": len(hol["days_without_fills"]), "max_gap_h": hol["max_gap_h"]}

    # 6 floor choice using only pre-test data (replay needs hourly PAXG: from 2020-10)
    tmp = ROOT / "state" / "_audit_cfg.json"
    sel = {}
    for fl in (0.05, 0.10):
        c = json.load(open(cfg)); c["risk_parity"]["trend_floor"] = fl; tmp.write_text(json.dumps(c))
        r = rp.run("2020-10-01", "2023-01-01", str(tmp), seed=0)
        sel[fl] = {"days_without_fill": len(r["days_without_fills"]), "days": r["days"], "max_gap_h": r["max_gap_h"],
                   **replay_metrics(r["equity"])}
    tmp.unlink(missing_ok=True)
    out["floor_choice_pretest"] = sel
    json.dump(out, open(ROOT / "results" / "final_audit.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=lambda x: round(float(x), 3)))


if __name__ == "__main__":
    main()
