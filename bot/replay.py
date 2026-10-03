"""Historical replay of the LIVE bot against a simulated Roostoo exchange.

Roostoo's API serves only live prices and orders (no history), so the bot cannot be run
over past years against the real exchange. This harness runs the unchanged live code
path - Bot.snapshot / Bot.plan (signal + order planner) and the Executor (limit-first
orders, 180 s timeout, market fallback, cash capping, Roostoo precision rules) - on a
simulated clock against FakeRoostoo, which answers with the same response schema the
real API returned on the test account (OrderDetail / OrderMatched, Status, Role,
FilledQuantity, CommissionChargeValue, ...).

Simulated market:
* prices: hourly Binance bars (data/full); inside an hour the price moves linearly from
  the bar's open to its close; bid/ask = price -/+ a half-spread (BTC 0.5 bp, ETH/PAXG 1 bp;
  measured Roostoo spreads for these pairs are ~0).
* LIMIT orders at the touch rest and fill as MAKER (0.05%) with a per-poll hazard
  calibrated so 89% fill within 180 s (16 of 18 on the real test account, 3 Oct 2026);
  MARKET orders fill at the far touch as TAKER (0.10%). Commission is charged in USD.
* the bot rebalances every `rebal_h` hours at hh:01 UTC, as the deployed config does.

    python3 -m bot.replay --start 2023-01-01 --end 2026-01-01
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bot.execution import Executor
from bot.live import Bot
from bot.strategies.live_rp import RiskParitySignal
from bot.strategy_config import load_strategy

ROOT = Path(__file__).resolve().parent.parent
RULES_FIXTURE = ROOT / "tests" / "fixtures_exchange_rules.json"
HALF_SPREAD = {"BTC/USD": 0.5e-4, "ETH/USD": 1e-4, "PAXG/USD": 1e-4}
MAKER, TAKER = 0.0005, 0.0010
P_FILL_180S = 16 / 18


class FakeRoostoo:
    env = "replay"

    def __init__(self, hourly: pd.DataFrame, clock, usd: float = 100_000.0, poll_s: float = 5.0, seed: int = 0):
        self.h = hourly                     # index: bar open time (UTC, tz-naive), columns: pairs, values (open, close)
        self.clock = clock
        self.wallet = {"USD": {"Free": usd, "Lock": 0.0}}
        self.orders: dict[int, dict] = {}
        self.next_id = 1
        self.rng = np.random.default_rng(seed)
        self.hazard = 1 - (1 - P_FILL_180S) ** (poll_s / 180.0)
        self.fees_paid = 0.0
        self.fills = {"MAKER": 0, "TAKER": 0}

    # ---------- market
    def price(self, pair: str) -> float:
        t = pd.Timestamp(self.clock(), unit="s")
        h0 = t.floor("h")
        o, c = self.h.at[h0, (pair, "open")], self.h.at[h0, (pair, "close")]
        return float(o + (c - o) * ((t - h0).total_seconds() / 3600.0))

    def ticker(self, pair: str | None = None) -> dict:
        pairs = [pair] if pair else list(HALF_SPREAD)
        out = {}
        for p in pairs:
            px = self.price(p); hs = HALF_SPREAD[p]
            out[p] = {"MaxBid": px * (1 - hs), "MinAsk": px * (1 + hs), "LastPrice": px, "Change": 0.0,
                      "CoinTradeValue": 0.0, "UnitTradeValue": 0.0}
        return out

    # ---------- account
    def balance(self) -> dict:
        return {k: dict(v) for k, v in self.wallet.items() if v["Free"] + v["Lock"] > 0 or k == "USD"}

    def short_positions(self) -> list:
        return []

    def _w(self, coin):
        return self.wallet.setdefault(coin, {"Free": 0.0, "Lock": 0.0})

    def _detail(self, o: dict) -> dict:
        return {"Pair": o["pair"], "OrderID": o["id"], "Status": o["status"], "Role": o["role"],
                "Side": o["side"], "Type": o["type"], "Price": o["price"], "Quantity": o["qty"],
                "FilledQuantity": o["qty"] if o["status"] in ("FILLED", "PENDING") else 0.0,  # Roostoo echoes qty while pending
                "FilledAverPrice": o["fill_px"], "CoinChange": o["qty"] if o["status"] == "FILLED" else 0.0,
                "UnitChange": o["qty"] * o["fill_px"], "CommissionCoin": "USD", "CommissionChargeValue": o["fee"],
                "CommissionPercent": MAKER if o["role"] == "MAKER" else TAKER}

    def _settle(self, o: dict, px: float, role: str) -> None:
        coin = o["pair"].split("/")[0]
        fee_rate = MAKER if role == "MAKER" else TAKER
        notional = o["qty"] * px
        fee = notional * fee_rate
        if o["side"] == "BUY":
            usd = self._w("USD")
            if o["type"] == "LIMIT":
                usd["Lock"] -= o["lock"]; usd["Free"] += o["lock"]
            usd["Free"] -= notional + fee
            self._w(coin)["Free"] += o["qty"]
        else:
            w = self._w(coin)
            if o["type"] == "LIMIT":
                w["Lock"] -= o["qty"]
            else:
                w["Free"] -= o["qty"]
            self._w("USD")["Free"] += notional - fee
        o.update(status="FILLED", role=role, fill_px=px, fee=fee)
        self.fees_paid += fee
        self.fills[role] += 1

    def place_order(self, pair, side, quantity, order_type="MARKET", price=None) -> dict:
        side, order_type, q = side.upper(), order_type.upper(), float(quantity)
        coin = pair.split("/")[0]
        o = {"id": self.next_id, "pair": pair, "side": side, "type": order_type, "qty": q,
             "price": float(price or 0.0), "status": "PENDING", "role": "MAKER" if order_type == "LIMIT" else "TAKER",
             "fill_px": 0.0, "fee": 0.0, "lock": 0.0}
        self.next_id += 1
        tick = self.ticker(pair)[pair]
        if side == "BUY":
            cost = q * (o["price"] if order_type == "LIMIT" else tick["MinAsk"]) * (1 + TAKER)
            if cost > self._w("USD")["Free"] + 1e-9:
                return {"Success": False, "ErrMsg": "insufficient balance"}
        elif q > self._w(coin)["Free"] + 1e-12:
            return {"Success": False, "ErrMsg": "insufficient balance"}
        if order_type == "MARKET":
            self._settle(o, tick["MinAsk"] if side == "BUY" else tick["MaxBid"], "TAKER")
        else:
            if side == "BUY":
                o["lock"] = q * o["price"] * (1 + MAKER)
                self._w("USD")["Free"] -= o["lock"]; self._w("USD")["Lock"] += o["lock"]
            else:
                self._w(coin)["Free"] -= q; self._w(coin)["Lock"] += q
        self.orders[o["id"]] = o
        return {"Success": True, "ErrMsg": "", "OrderDetail": self._detail(o)}

    def query_order(self, order_id=None, pair=None, pending_only=None, offset=None, limit=None) -> dict:
        o = self.orders[int(order_id)]
        if o["status"] == "PENDING" and self.rng.random() < self.hazard:
            self._settle(o, o["price"], "MAKER")
        return {"Success": True, "ErrMsg": "", "OrderMatched": [self._detail(o)]}

    def cancel_order(self, order_id=None, pair=None) -> dict:
        o = self.orders[int(order_id)]
        if o["status"] == "PENDING":
            coin = o["pair"].split("/")[0]
            if o["side"] == "BUY":
                u = self._w("USD"); u["Lock"] -= o["lock"]; u["Free"] += o["lock"]
            else:
                w = self._w(coin); w["Lock"] -= o["qty"]; w["Free"] += o["qty"]
            o["status"] = "CANCELED"
        return {"Success": True, "ErrMsg": "", "CanceledList": [o["id"]]}

    def equity(self) -> float:
        tick = self.ticker()
        eq = self.wallet["USD"]["Free"] + self.wallet["USD"]["Lock"]
        for coin, b in self.wallet.items():
            if coin != "USD":
                eq += (b["Free"] + b["Lock"]) * tick[f"{coin}/USD"]["MaxBid"]
        return eq


def load_hourly() -> pd.DataFrame:
    cols = {}
    for coin in ("BTC", "ETH", "PAXG"):
        d = pd.read_csv(ROOT / "data" / "full" / f"{coin}.csv", index_col=0, parse_dates=True)
        d.index = d.index.tz_convert(None)
        d = d[~d.index.duplicated()].asfreq("h").ffill()
        cols[(f"{coin}/USD", "open")] = d["open"]
        cols[(f"{coin}/USD", "close")] = d["close"]
    return pd.DataFrame(cols)          # keep each asset's full history (no cross-asset truncation)


def run(start: str, end: str, config: str, seed: int = 0, quiet: bool = True) -> dict:
    logging.basicConfig(level=logging.WARNING if quiet else logging.INFO)
    cfg = load_strategy(config)
    rules = json.loads(RULES_FIXTURE.read_text())
    H = load_hourly()
    daily = pd.DataFrame({c: H[(f"{c}/USD", "close")].resample("1D").last() for c in ("BTC", "ETH", "PAXG")})
    sim = {"t": pd.Timestamp(start).timestamp() + 60}
    clock = lambda: sim["t"]                                            # noqa: E731

    def sleep(s):
        sim["t"] += max(float(s), 0.0)
    ex_client = FakeRoostoo(H, clock, poll_s=cfg.execution.poll_s, seed=seed)
    journal = ROOT / "logs" / "replay_trades.jsonl"
    journal.unlink(missing_ok=True)
    ex = Executor(ex_client, rules, "replay", cfg.execution, dry_run=False, log_path=journal, now=clock, sleep=sleep)
    bot = Bot(ex_client, cfg, None, ex, dry_run=False, rules=rules)
    sig = RiskParitySignal(cfg.risk_parity, fetch=None)
    bot.sig = sig
    bot.state = {}                                                      # fresh account state for every replay
    days = pd.date_range(start, end, freq="D", inclusive="left")
    step_h = cfg.rebal_h
    eq_rows, legs, fill_days, fill_ts = [], [], set(), []
    for D in days:
        P = daily[daily.index < D].iloc[-600:]                          # completed daily bars only
        sig._day, sig._P = pd.Timestamp.now(tz="UTC").normalize(), P    # bypass the live date filter
        for h in range(0, 24, step_h):
            t0 = D.timestamp() + h * 3600 + 60                          # hh:01 UTC
            sim["t"] = t0
            s = bot.snapshot()
            ex.set_plan(bot.plan(s, 1.0, now_ts=t0))
            ex.tick(t0 + step_h * 3600 - 600)                           # work the queue until 10 min before the next slot
            if ex.busy:
                ex.cancel_all("slot end")
            n0 = len(legs)
            legs += ex.done
            if any(str(l["outcome"]).startswith("filled") for l in ex.done):
                fill_days.add(D)
                fill_ts.append(t0)
            bot.note_fills(now_ts=sim["t"])
            ex.done = []; bot._fills_seen = 0
        sim["t"] = (D + pd.Timedelta(days=1)).timestamp() - 1           # mark at the next daily close
        eq_rows.append((D, ex_client.equity()))
    eq = pd.Series(dict(eq_rows)).sort_index()
    outcomes = pd.Series([l["outcome"] for l in legs]).value_counts().to_dict()
    return {"equity": eq, "fees": ex_client.fees_paid, "fills": ex_client.fills, "outcomes": outcomes,
            "orders": ex_client.next_id - 1, "days": len(days), "days_with_fills": len(fill_days),
            "days_without_fills": sorted(str(d.date()) for d in set(days) - fill_days),
            "fill_ts": fill_ts,
            "max_gap_h": float(np.diff([days[0].timestamp()] + fill_ts + [days[-1].timestamp() + 86400]).max() / 3600)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2026-01-01")
    ap.add_argument("--config", default=str(ROOT / "config" / "strategy.json"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    res = run(a.start, a.end, a.config, a.seed)
    eq = res["equity"]
    eq.to_csv(ROOT / "results" / f"replay_equity_{a.start}_{a.end}.csv")
    print(json.dumps({k: v for k, v in res.items() if k != "equity"}, default=str))
    print(f"final equity {eq.iloc[-1]:,.0f} from 100,000")


if __name__ == "__main__":
    main()
