"""Live adapter for the risk-balanced trend portfolio (bot.strategies.risk_parity).

Fetches completed daily Binance bars for each sleeve (BTCUSDT, ETHUSDT, PAXGUSDT),
computes the weights with the same `target_weights` the backtest uses, and returns the
last row keyed by Roostoo pair. Bars are cached per UTC day.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from bot.data import fetch_klines
from bot.strategies.risk_parity import ANN, RPParams, target_weights

log = logging.getLogger("signal")
HISTORY_DAYS = 600   # >= 200d SMA + EWMA burn-in (halflife 30d -> 600d leaves < 1e-6 weight)


class RiskParitySignal:
    def __init__(self, p: RPParams, fetch=fetch_klines):
        self.p, self.fetch = p.validate(), fetch
        self._day, self._P = None, None
        log.info("signal: risk_parity %s", p)

    def daily_closes(self) -> pd.DataFrame:
        today = pd.Timestamp.now(tz="UTC").normalize()
        if self._day == today and self._P is not None:
            return self._P
        cols = {}
        for c in self.p.sleeves:
            df = self.fetch(c + "USDT", "1d", days=HISTORY_DAYS)
            if df is None or df.empty:
                raise RuntimeError(f"no daily bars for {c}")
            df = df[df.index < today]                     # drop today's still-forming bar
            cols[c] = df["close"]
        P = pd.DataFrame(cols).sort_index()
        self._day, self._P = today, P
        return P

    def target_weights(self, coins: list[str] | None = None) -> tuple[pd.Series, dict]:
        P = self.daily_closes()
        W = target_weights(P, self.p)
        w = W.iloc[-1]
        R = P.pct_change(fill_method=None)
        vol = np.sqrt((R ** 2).ewm(halflife=self.p.halflife, min_periods=20).mean()).iloc[-1] * np.sqrt(ANN)
        sma = {L: P.rolling(L).mean().iloc[-1] for L in self.p.lookbacks}
        diag = {"kind": "risk_parity", "bar": str(P.index[-1].date()),
                "trend_on": {c: [bool(P[c].iloc[-1] > sma[L][c]) for L in self.p.lookbacks] for c in self.p.sleeves},
                "vol_ann": {c: round(float(vol[c]), 3) for c in self.p.sleeves},
                "weights": {c: round(float(w[c]), 4) for c in self.p.sleeves},
                "gross": round(float(w.sum()), 3), "target_vol": self.p.target_vol}
        out = pd.Series(w.values, index=[c + "/USD" for c in self.p.sleeves])
        return out, diag
