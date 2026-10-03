"""Bias audit of the Sharpe-first strategy (bot/strategies/risk_parity.py).

1. Causality: weights at day T computed on data truncated at T equal the full-history
   weights at T (no future data in any weight).
2. Timing / execution: extra 1-day lag; execution 1-6 hours after the daily close using
   hourly prices (live trades a few minutes to an hour after 00:00 UTC).
3. Costs: 2x and 3x fees.
4. Selection: is the volatility target choosable from train+val alone? Distribution of
   test Sharpe over ALL grid configs (no selection) and over random-signal placebos.
5. Data: drop the partial last day; GLD-only gold instead of the GLD/PAXG splice.

    python3 -m bot.research.sharpe_audit
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot.research.sharpe_opt import SPLITS, load_prices, metrics, run, split
from bot.strategies.risk_parity import RPParams, simulate, target_weights

SL, BUD, LB, HL = ["BTC", "ETH", "GOLD"], [2.0, 2.0, 1.0], [200], 30
RNG = np.random.default_rng(7)


def sharpe_by_split(r):
    return {sp: round(metrics(split(r, sp)).get("Sharpe", np.nan), 2) for sp in SPLITS}


def main():
    P = load_prices().iloc[:-1]                       # drop the partial last day
    p = RPParams(sleeves=SL, budgets=BUD, lookbacks=LB, halflife=HL, target_vol=0.20)
    W = target_weights(P, p)
    base = simulate(P, W, 0.02, 0.001)
    print("baseline (tv 0.20):", sharpe_by_split(base))

    # 1. causality
    bad = 0
    for T in RNG.choice(np.arange(400, len(P)), 25, replace=False):
        wt = target_weights(P.iloc[: T + 1], p).iloc[-1]
        bad += not np.allclose(wt.values, W.iloc[T].values, atol=1e-6)
    print(f"1. causality: {25 - bad}/25 truncation checks identical")

    # 2a. extra one-day lag
    print("2a. weights lagged one extra day:", sharpe_by_split(simulate(P, W.shift(1).fillna(0), 0.02, 0.001)))
    # 2b. execution h hours after the daily close, hourly marking (2020-09 onward: PAXG hourly exists)
    H = {}
    for c, f in (("BTC", "BTC"), ("ETH", "ETH"), ("GOLD", "PAXG")):
        s = pd.read_csv(f"data/full/{f}.csv", index_col=0, parse_dates=True)["close"]
        s.index = s.index.tz_convert(None) + pd.Timedelta(hours=1)    # label by close time
        H[c] = s
    H = pd.DataFrame(H)
    H = H[(H.index >= "2020-10-01") & (H.index < P.index[-1])].ffill()
    Rh = H.pct_change().fillna(0.0)
    for lag_h in (0, 1, 3, 6):
        # daily weight decided at close of day D (= 00:00 of D+1), effective from 00:00+lag_h
        Wd = W.copy(); Wd.index = Wd.index + pd.Timedelta(days=1) + pd.Timedelta(hours=lag_h)
        Wh = Wd.reindex(H.index, method="ffill").fillna(0.0)
        cur = np.zeros(3); eq = 1.0; out = []
        rv, wv = Rh.values, Wh.values
        for i in range(len(H)):
            pr = float(cur @ rv[i]); eq *= 1 + pr
            if 1 + pr > 0:
                cur = cur * (1 + rv[i]) / (1 + pr)
            diff = wv[i] - cur
            tr = np.where(np.abs(diff) > 0.02, diff, 0.0); tr = np.where((wv[i] == 0) & (cur != 0), -cur, tr)
            to = np.abs(tr).sum()
            if to:
                eq *= 1 - to * 0.001; cur = cur + tr
            out.append(eq)
        d = pd.Series(out, index=H.index).resample("1D").last().pct_change().dropna()
        print(f"2b. execute {lag_h}h after close (hourly marking): test Sharpe {metrics(split(d, 'test'))['Sharpe']:.2f}"
              f"  holdout {metrics(split(d, 'holdout'))['Sharpe']:.2f}")

    # 3. costs
    for fee in (0.002, 0.003):
        print(f"3. fee {fee:.1%}:", sharpe_by_split(simulate(P, W, 0.02, fee)))
    to = (W.diff().abs().sum(1)).groupby(W.index.year).sum()
    print("   turnover per year (x equity):", to.round(1).to_dict())

    # 4a. vol target chosen on train+val only
    print("4a. vol-target choice from train+val only:")
    for tv in (0.10, 0.15, 0.20, 0.25, 0.30):
        r, _ = run(P, SL, BUD, LB, HL, tv)
        tr_ = [metrics(split(r, s)) for s in ("train", "val")]
        trv = r[(r.index >= SPLITS["train"][0]) & (r.index < SPLITS["val"][1])]
        eq = (1 + trv).cumprod(); w = [eq[s: s + pd.Timedelta(days=14)] for s in trv.index[:-14:3]]
        w14 = np.array([x.iloc[-1] / x.iloc[0] - 1 for x in w])
        print(f"   tv {tv:.2f}: train+val composite {np.mean([m['composite'] for m in tr_]):.2f}"
              f"  Sharpe train {tr_[0]['Sharpe']:.2f} val {tr_[1]['Sharpe']:.2f}"
              f"  | 14d (train+val) median {np.median(w14):+.2%} P(>3%) {np.mean(w14 > 0.03):.2f}")
    # 4b. all grid configs, no selection
    g = pd.read_csv("results/sharpe_grid.csv"); g = g[g.budget != "-"]
    print(f"4b. ALL {len(g)} grid configs, test Sharpe: median {g.test_Sharpe.median():.2f}, "
          f"10th pct {g.test_Sharpe.quantile(.1):.2f}, share > BTC hold (1.42) {np.mean(g.test_Sharpe > 1.42):.2f}")
    # 4c. placebo: identical sizing machinery (risk parity + vol target), random persistent
    #     on/off signals with the same average holding period as the 200d filter
    on = trend_signal_mean = W.gt(0).mean()
    switches = (W.gt(0).astype(int).diff().abs().sum() / len(W)).clip(lower=1 / 365)
    pl = []
    for _ in range(300):
        sig = pd.DataFrame({c: (pd.Series(RNG.random(len(P)) < switches[c]).cumsum() % 2).values
                            for c in SL}, index=P.index).astype(float)
        rp = simulate(P, target_weights(P, p, signal=sig), 0.02, 0.001)
        pl.append(metrics(split(rp, "test"))["Sharpe"])
    pl = np.array(pl)
    st = metrics(split(base, "test"))["Sharpe"]
    print(f"4c. random-signal placebo (300 runs, same vol targeting) test Sharpe: median {np.median(pl):.2f},"
          f" 95th pct {np.percentile(pl, 95):.2f}; strategy {st:.2f} -> p = {np.mean(pl >= st):.3f}")
    vt = simulate(P, target_weights(P, p, trend=False), 0.02, 0.001)
    print("    vol targeting alone, no trend filter (always on):", sharpe_by_split(vt))

    # 5. gold data: GLD only (no PAXG splice)
    P2 = P.copy()
    gld = pd.read_csv("data/stocks/GLD.csv", index_col=0, parse_dates=True)["close"]
    P2["GOLD"] = gld.reindex(P2.index).ffill()
    print("5. GLD-only gold:", sharpe_by_split(simulate(P2, target_weights(P2, p), 0.02, 0.001)))


if __name__ == "__main__":
    main()
