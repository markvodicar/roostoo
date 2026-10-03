"""Order planning and limit-first execution against the Roostoo API.

plan_orders()  pure function: target USD exposures + current positions -> list of
               Orders, closes/sells first (they free USD), then buys/short opens,
               largest first; quantities floored to AmountPrecision; legs below
               MiniOrder / min_trade_usd dropped; drifts inside the band ignored
               unless the target is exactly zero (full exit).

Executor       state machine driven by Bot.step() every cycle via tick(until):
               * spot legs: post a LIMIT at the touch (buy at MaxBid, sell at MinAsk,
                 rounded to PricePrecision) -> maker fee 0.05%; poll query_order every
                 poll_s; after limit_timeout_s cancel and send the remainder as MARKET
               * shorts (v6, collateral in USD) are market only
               * at most one new order per order_gap_s (competition pacing)
               * buys / short opens are re-capped against the live free USD right
                 before sending (fees included), so a partially filled sell cannot
                 make a later buy fail
               * every request and raw response is appended to logs/trades_<env>.jsonl
               * dry_run: nothing is sent; legs are logged and treated as filled

Roostoo response conventions discovered on the test account (2026-10-03):
  query_order -> {"OrderMatched": [{OrderID, Status: FILLED|CANCELED|PENDING..., Role:
  MAKER|TAKER, Quantity, FilledQuantity, FilledAverPrice, CoinChange, UnitChange,
  CommissionCoin, CommissionChargeValue, CommissionPercent, ...}]}. For a CANCELED
  order FilledQuantity echoes Quantity while CoinChange is the real fill, so the
  executed amount is taken from FilledQuantity only when Status == FILLED and from
  |CoinChange| otherwise. short_positions -> {"Positions": [...]}. pending_count
  raises "no pending order under this account" when nothing is open.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from bot.client import RoostooError
from bot.strategy_config import ExecutionParams

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
log = logging.getLogger("exec")
TERMINAL_OK = {"FILLED"}
TERMINAL_GONE = {"CANCELED", "CANCELLED", "REJECTED", "EXPIRED", "FAILED"}
KIND_RANK = {"short_close": 0, "sell": 1, "buy": 2, "short_open": 3}


@dataclass
class Order:
    kind: str                 # buy | sell | short_open | short_close
    pair: str
    qty: float = 0.0          # coins (buy / sell)
    usd: float = 0.0          # notional estimate at planning time
    collateral: float = 0.0   # USD collateral (short_open)
    pct: float = 100.0        # share of the short to close (short_close)
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------- rounding
def floor_qty(q: float, rule: dict) -> float:
    f = 10 ** int(rule.get("AmountPrecision", 8))
    return math.floor(q * f + 1e-9) / f


def round_price(px: float, rule: dict) -> float:
    return round(float(px), int(rule.get("PricePrecision", 8)))


def min_notional(rule: dict, min_trade_usd: float) -> float:
    return max(float(rule.get("MiniOrder", 0) or 0), min_trade_usd)


# ----------------------------------------------------------------------------- planning
def plan_orders(targets_usd: dict, positions_usd: dict, free_qty: dict, prices: dict, rules: dict,
                band_usd: float, min_trade_usd: float = 25.0) -> list[Order]:
    """targets_usd/positions_usd: pair -> signed USD (short < 0); free_qty: pair -> free
    coins; prices: pair -> reference price (last). Pairs with a position but no target
    are liquidated (target 0). Returns closes/sells first, then buys/short opens."""
    orders: list[Order] = []
    pairs = sorted(set(targets_usd) | set(positions_usd))
    for pair in pairs:
        rule = rules.get(pair)
        px = prices.get(pair)
        t, c = float(targets_usd.get(pair, 0.0)), float(positions_usd.get(pair, 0.0))
        if rule is None or not px or px <= 0 or not rule.get("CanTrade", True):
            if abs(t) > 0 or abs(c) > 0:
                log.warning("plan: skipping %s (no rule/price or not tradeable)", pair)
            continue
        full_exit = t == 0.0 and c != 0.0
        flip = t * c < 0.0                       # long -> short or short -> long: always trade
        if abs(t - c) < max(band_usd, min_trade_usd) and not (full_exit or flip):
            continue
        floor_usd = min_notional(rule, min_trade_usd)
        # 1) unwind anything on the wrong side or oversized
        if c < 0 and (t >= 0 or abs(t) < abs(c)):
            pct = 100.0 if t >= 0 else round((abs(c) - abs(t)) / abs(c) * 100, 2)
            orders.append(Order("short_close", pair, pct=pct, usd=abs(c) * pct / 100, reason="reduce short"))
            c = 0.0 if t >= 0 else t
        if c > 0 and (t <= 0 or t < c):
            sell_usd = c if t <= 0 else c - t
            q = floor_qty(min(float(free_qty.get(pair, 0.0)), sell_usd / px), rule)
            if q > 0 and q * px >= float(rule.get("MiniOrder", 0) or 0):
                orders.append(Order("sell", pair, qty=q, usd=q * px, reason="exit" if t <= 0 else "reduce"))
            c = 0.0 if t <= 0 else t
        # 2) build toward target
        if t > c and t > 0:
            q = floor_qty((t - c) / px, rule)
            if q > 0 and q * px >= floor_usd:
                orders.append(Order("buy", pair, qty=q, usd=q * px, reason="enter" if c == 0 else "add"))
        elif t < c and t < 0:
            coll = round(abs(t) - abs(min(c, 0.0)), 2)
            if coll >= floor_usd:
                orders.append(Order("short_open", pair, collateral=coll, usd=coll, reason="short"))
    orders.sort(key=lambda o: (KIND_RANK[o.kind], -o.usd))
    return orders


def cap_to_cash(o: Order, usd_free: float, price: float, rule: dict, xp: ExecutionParams) -> Order | None:
    """Shrink a buy / short_open to the USD actually available (fees included).
    Returns None when the capped leg is below MiniOrder / min_trade_usd."""
    cash = max(usd_free, 0.0) * xp.fee_buffer
    floor_usd = min_notional(rule, xp.min_trade_usd)
    if o.kind == "buy":
        q = min(o.qty, floor_qty(cash / price, rule))
        if q <= 0 or q * price < floor_usd:
            return None
        return Order(o.kind, o.pair, qty=q, usd=q * price, reason=o.reason)
    if o.kind == "short_open":
        coll = round(min(o.collateral, cash), 2)
        if coll < floor_usd:
            return None
        return Order(o.kind, o.pair, collateral=coll, usd=coll, reason=o.reason)
    return o


def extract_order_id(resp) -> int | None:
    """Roostoo nests the order under different keys; find the first OrderID anywhere."""
    if isinstance(resp, dict):
        for k, v in resp.items():
            if k.lower() in ("orderid", "order_id") and v not in (None, ""):
                try:
                    return int(v)
                except (TypeError, ValueError):
                    return None
        for v in resp.values():
            r = extract_order_id(v)
            if r is not None:
                return r
    elif isinstance(resp, list):
        for v in resp:
            r = extract_order_id(v)
            if r is not None:
                return r
    return None


def order_status(resp: dict, order_id: int) -> dict | None:
    rows = resp.get("OrderMatched") if isinstance(resp, dict) else None
    if not rows and isinstance(resp, dict):
        rows = resp.get("OrderDetail") or resp.get("Orders")
        rows = [rows] if isinstance(rows, dict) else rows
    for r in rows or []:
        if isinstance(r, dict) and str(r.get("OrderID")) == str(order_id):
            return r
    return None


def filled_qty(row: dict) -> float:
    st = str(row.get("Status", "")).upper()
    if st in TERMINAL_OK:
        return float(row.get("FilledQuantity") or row.get("Quantity") or 0.0)
    return abs(float(row.get("CoinChange") or 0.0))


# ----------------------------------------------------------------------------- executor
class Executor:
    def __init__(self, client, rules: dict, env: str, xp: ExecutionParams, dry_run: bool = False,
                 log_path: Path | None = None, now=time.time, sleep=time.sleep):
        self.c, self.rules, self.xp, self.dry = client, rules, xp, dry_run
        self.env = env.lower()
        self.log_path = log_path or (LOG_DIR / f"trades_{self.env}.jsonl")
        self.now, self.sleep = now, sleep
        self.queue: list[Order] = []
        self.active: dict | None = None     # resting limit: {order, order_id, price, qty, placed_at}
        self.last_order_ts = 0.0
        self.done: list[dict] = []          # completed legs this session (for logging / report)

    # -- journaling
    def journal(self, event: str, **kw) -> None:
        rec = {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "env": self.env,
               "dry_run": self.dry, "event": event, **kw}
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except OSError as e:
            log.error("trade journal write failed: %s", e)

    def _call(self, name: str, order: Order, **params):
        """Invoke a client method, journal request + raw response (or error), re-raise."""
        fn = getattr(self.c, name)
        try:
            resp = fn(**params)
        except RoostooError as e:
            self.journal(name, order=order.to_dict(), request=params, error=str(e))
            raise
        self.journal(name, order=order.to_dict(), request=params, response=resp)
        return resp

    # -- plan management
    def set_plan(self, orders: list[Order]) -> None:
        if self.active is not None:
            self._cancel_active("replanned")
        self.queue = list(orders)

    def cancel_all(self, reason: str = "cancel_all") -> None:
        if self.active is not None:
            self._cancel_active(reason)
        self.queue.clear()

    @property
    def busy(self) -> bool:
        return bool(self.queue) or self.active is not None

    def status(self) -> str:
        a = self.active
        return f"queue {len(self.queue)}" + (f", resting {a['order'].kind} {a['order'].pair} #{a['order_id']}" if a else "")

    # -- market data helpers
    def _touch(self, pair: str) -> tuple[float, float, float]:
        t = self.c.ticker(pair)
        row = t[pair] if isinstance(t, dict) and pair in t else t
        return float(row["MaxBid"]), float(row["MinAsk"]), float(row["LastPrice"])

    def _usd_free(self) -> float:
        bal = self.c.balance()
        return float((bal.get("USD") or {}).get("Free", 0.0))

    def _wait_gap(self, until: float) -> bool:
        """Sleep until a new order is allowed. False if that would pass `until`."""
        wait = self.last_order_ts + self.xp.order_gap_s - self.now()
        if wait <= 0:
            return True
        if self.now() + wait > until:
            return False
        self.sleep(wait)
        return True

    # -- main entry point
    def tick(self, until: float) -> None:
        """Work the queue until `until` (epoch seconds). Never raises."""
        while self.now() < until:
            try:
                if self.active is not None:
                    if not self._service_active(until):
                        return
                    continue
                if not self.queue:
                    return
                if not self._wait_gap(until):
                    return
                o = self.queue.pop(0)
                self._start(o)
            except RoostooError as e:
                log.error("execution error: %s", e)
                self.sleep(min(self.xp.poll_s, max(until - self.now(), 0)))
            except Exception:  # noqa: BLE001 - keep the bot alive
                log.exception("unexpected execution error")
                self.sleep(min(self.xp.poll_s, max(until - self.now(), 0)))

    # -- one leg
    def _start(self, o: Order) -> None:
        rule = self.rules.get(o.pair, {})
        if o.kind in ("buy", "sell"):
            bid, ask, last = self._touch(o.pair)
            if o.kind == "buy":
                capped = cap_to_cash(o, self._usd_free(), ask, rule, self.xp)
                if capped is None:
                    self._finish(o, "skipped", note="insufficient USD")
                    return
                o = capped
            px = round_price(bid if o.kind == "buy" else ask, rule)
            self._place_limit(o, px)
        elif o.kind == "short_open":
            capped = cap_to_cash(o, self._usd_free(), 1.0, rule, self.xp)
            if capped is None:
                self._finish(o, "skipped", note="insufficient USD")
                return
            self._send_market(capped)
        else:
            self._send_market(o)

    def _place_limit(self, o: Order, px: float) -> None:
        self.last_order_ts = self.now()
        if self.dry:
            log.info("[DRY] LIMIT %s %s %s @ %s (~%.0f USD)", o.kind.upper(), o.qty, o.pair, px, o.qty * px)
            self.journal("dry_limit", order=o.to_dict(), price=px)
            self._finish(o, "filled_dry", price=px, qty=o.qty)
            return
        resp = self._call("place_order", o, pair=o.pair, side=o.kind.upper(), quantity=o.qty,
                          order_type="LIMIT", price=px)
        oid = extract_order_id(resp)
        log.info("LIMIT %s %s %s @ %s -> id %s", o.kind.upper(), o.qty, o.pair, px, oid)
        if oid is None:
            log.error("no OrderID in place_order response: %s", json.dumps(resp)[:300])
            self._finish(o, "error", note="no order id", response=resp)
            return
        self.active = {"order": o, "order_id": oid, "price": px, "qty": o.qty, "placed_at": self.now()}

    def _service_active(self, until: float) -> bool:
        """Poll / time out the resting limit. Returns False when tick() should return."""
        a = self.active
        o: Order = a["order"]
        resp = self._call("query_order", o, order_id=a["order_id"])
        row = order_status(resp, a["order_id"])
        st = str(row.get("Status", "")).upper() if row else "UNKNOWN"
        filled = filled_qty(row) if row else 0.0
        if st in TERMINAL_OK:
            self.active = None
            self._finish(o, "filled_limit", qty=filled, price=row.get("FilledAverPrice"), role=row.get("Role"),
                         fee=row.get("CommissionChargeValue"), fee_coin=row.get("CommissionCoin"), order_id=a["order_id"])
            return True
        if st in TERMINAL_GONE:
            self.active = None
            self._market_remainder(o, a["qty"] - filled, "limit gone", a["order_id"], filled)
            return True
        if self.now() - a["placed_at"] >= self.xp.limit_timeout_s:
            self.active = None
            filled = self._cancel_active_order(o, a["order_id"], filled)
            self._market_remainder(o, a["qty"] - filled, "limit timeout", a["order_id"], filled)
            return True
        wait = min(self.xp.poll_s, until - self.now())
        if wait <= 0:
            return False
        self.sleep(wait)
        return True

    def _cancel_active_order(self, o: Order, oid: int, filled: float) -> float:
        try:
            self._call("cancel_order", o, order_id=oid)
        except RoostooError as e:  # already filled between poll and cancel is the common cause
            log.warning("cancel %s failed: %s", oid, e)
        try:
            resp = self._call("query_order", o, order_id=oid)
            row = order_status(resp, oid)
            if row:
                st = str(row.get("Status", "")).upper()
                filled = filled_qty(row)
                if st in TERMINAL_OK:
                    self._finish(o, "filled_limit", qty=filled, price=row.get("FilledAverPrice"), role=row.get("Role"),
                                 fee=row.get("CommissionChargeValue"), fee_coin=row.get("CommissionCoin"), order_id=oid)
                    return filled
        except RoostooError as e:
            log.warning("post-cancel query %s failed: %s", oid, e)
        if filled > 0:
            self.journal("partial_fill", order=o.to_dict(), order_id=oid, filled=filled)
        return filled

    def _cancel_active(self, reason: str) -> None:
        a, self.active = self.active, None
        if a is None:
            return
        o: Order = a["order"]
        try:
            self._call("cancel_order", o, order_id=a["order_id"])
            log.info("cancelled resting %s %s (%s)", o.kind, o.pair, reason)
        except RoostooError as e:
            log.warning("cancel %s failed (%s): %s", a["order_id"], reason, e)
        self._finish(o, "cancelled", note=reason, order_id=a["order_id"])

    def _market_remainder(self, o: Order, remaining: float, why: str, oid: int, filled: float) -> None:
        rule = self.rules.get(o.pair, {})
        q = floor_qty(max(remaining, 0.0), rule)
        _, _, last = self._touch(o.pair)
        if q <= 0 or q * last < float(rule.get("MiniOrder", 0) or 0) or q * last < 1.0:
            self._finish(o, "done_after_limit", note=f"{why}; remainder {q} too small", filled_limit=filled, order_id=oid)
            return
        rem = Order(o.kind, o.pair, qty=q, usd=q * last, reason=f"{o.reason} ({why})")
        if not self._wait_gap(float("inf")):
            return
        self._send_market(rem)

    def _send_market(self, o: Order) -> None:
        self.last_order_ts = self.now()
        if self.dry:
            log.info("[DRY] MARKET %s", o)
            self.journal("dry_market", order=o.to_dict())
            self._finish(o, "filled_dry")
            return
        if o.kind in ("buy", "sell"):
            resp = self._call("place_order", o, pair=o.pair, side=o.kind.upper(), quantity=o.qty, order_type="MARKET")
            oid = extract_order_id(resp)
            row = None
            if oid is not None:
                try:
                    row = order_status(self._call("query_order", o, order_id=oid), oid)
                except RoostooError as e:
                    log.warning("query after market order failed: %s", e)
            log.info("MARKET %s %s %s -> id %s status %s", o.kind.upper(), o.qty, o.pair, oid, row.get("Status") if row else "?")
            self._finish(o, "filled_market", order_id=oid, qty=filled_qty(row) if row else o.qty,
                         price=(row or {}).get("FilledAverPrice"), role=(row or {}).get("Role"),
                         fee=(row or {}).get("CommissionChargeValue"), fee_coin=(row or {}).get("CommissionCoin"))
        elif o.kind == "short_open":
            resp = self._call("short_open", o, pair=o.pair, collateral_usd=o.collateral)
            log.info("SHORT OPEN %s collateral %.2f -> %s", o.pair, o.collateral, json.dumps(resp)[:200])
            self._finish(o, "short_opened", response=resp)
        elif o.kind == "short_close":
            resp = self._call("short_close", o, pair=o.pair, close_pct=None if o.pct >= 100 else o.pct)
            log.info("SHORT CLOSE %s %.1f%% -> %s", o.pair, o.pct, json.dumps(resp)[:200])
            self._finish(o, "short_closed", response=resp)

    def _finish(self, o: Order, outcome: str, **info) -> None:
        rec = {"order": o.to_dict(), "outcome": outcome, **info}
        self.done.append(rec)
        self.journal("leg_done", **rec)
        log.info("leg %s %s %s -> %s %s", o.kind, o.pair, o.qty or o.collateral or o.pct, outcome,
                 {k: v for k, v in info.items() if k in ("qty", "price", "role", "fee", "note")})
