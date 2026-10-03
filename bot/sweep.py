"""Compare strategy variants with an honest train/test split.

Train = older ~2/3 of the history, Test = most recent ~1/3 (never used to choose).
We rank on the TRAIN median 14-day competition score, then report TEST.
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import requests

from bot.backtest import metrics, run, window_report
from bot.data import load_panel
from bot.risk import RiskParams
from bot.strategy import Params

BASE = Params()
VARIANTS = {
    "base_hourly": (BASE, {}),
    "slow_lookbacks": (replace(BASE, lookbacks=(72, 168, 336), trend_ema=168, regime_ema=336), {}),
    "slow_top3": (replace(BASE, lookbacks=(72, 168, 336), trend_ema=168, regime_ema=336, top_k=3), {}),
    "slow_band6": (replace(BASE, lookbacks=(72, 168, 336), trend_ema=168, regime_ema=336), {"rebalance_band": 0.06}),
    "slow_lowvol": (replace(BASE, lookbacks=(72, 168, 336), trend_ema=168, regime_ema=336, target_vol=0.25), {"rebalance_band": 0.06}),
    "majors_only": (replace(BASE, lookbacks=(72, 168, 336), trend_ema=168, regime_ema=336, top_k=3), {"universe": "majors"}),
}
MAJORS = ["BTC/USD", "ETH/USD", "SOL/USD", "BNB/USD", "XRP/USD", "DOGE/USD", "ADA/USD", "LINK/USD",
          "AVAX/USD", "LTC/USD", "TRX/USD", "SUI/USD", "DOT/USD", "NEAR/USD", "TON/USD"]


def main():
    info = requests.get("https://mock-api.roostoo.com/v3/exchangeInfo", timeout=10).json()
    pairs = sorted(k for k, v in info["TradePairs"].items() if v.get("AssetType") == "crypto")
    closes = load_panel(pairs)
    split = closes.index[0] + (closes.index[-1] - closes.index[0]) * 2 / 3
    print(f"{closes.shape[1]} pairs, {closes.index[0]:%Y-%m-%d} .. {closes.index[-1]:%Y-%m-%d}, test from {split:%Y-%m-%d}")
    rows = []
    for name, (p, kw) in VARIANTS.items():
        kw = dict(kw)
        data = closes[[c for c in MAJORS if c in closes]] if kw.pop("universe", None) == "majors" else closes
        eq, trades = run(data, p, RiskParams(), **kw)
        for part, seg in (("train", eq[eq.index < split]), ("test", eq[eq.index >= split])):
            wr = window_report(seg)
            m = metrics(seg)
            rows.append({"variant": name, "part": part, "ret": m["return"], "maxdd": m["max_dd"],
                         "score_full": m["score"], "win_med_ret": wr["return"].median(),
                         "win_med_score": wr["score"].median(), "win_pos": (wr["return"] > 0).mean(),
                         "trades": len(trades)})
    btc = closes["BTC/USD"].dropna()
    for part, seg in (("train", btc[btc.index < split]), ("test", btc[btc.index >= split])):
        wr = window_report(seg / seg.iloc[0])
        m = metrics(seg / seg.iloc[0])
        rows.append({"variant": "BTC_hold", "part": part, "ret": m["return"], "maxdd": m["max_dd"],
                     "score_full": m["score"], "win_med_ret": wr["return"].median(),
                     "win_med_score": wr["score"].median(), "win_pos": (wr["return"] > 0).mean(), "trades": 0})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.pivot(index="variant", columns="part").round(3).to_string())


if __name__ == "__main__":
    main()
