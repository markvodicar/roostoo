"""Live trading loop for Roostoo (core + tilt strategy from config/strategy.json).

Every cycle (60s):
  1. reconcile: /v3/ticker (all pairs), /v3/balance, /v6/short_positions -> mark the
     book to market (longs at bid, shorts at ask) -> logs/equity_<env>.csv
  2. drawdown throttle (bot/risk.py): exposure scale from the 14-day rolling peak of
     hourly equity marks, persisted in state/state_<env>.json
  3. every 5 minutes: append every pair's MaxBid/MinAsk/LastPrice to
     data/ticker/<date>.csv (builds a dataset for the tokenised stocks)
  4. on the hour, every `rebal_h` hours (or when the throttle tightens by > 0.1):
     recompute target weights (bot.ml.live_signal) and hand the order plan to the
     limit-first executor (bot.execution)
  5. the executor works its queue for the rest of the cycle (one new order per 61s)

Kill switches (files in the repo root): STOP pauses rebalancing and cancels queued
orders (positions untouched); LIQUIDATE sells / closes everything, then pauses.
Every exchange call is made by this process and journaled to logs/trades_<env>.jsonl.

Usage:
  python3 -m bot.live --dry-run --once      # plan only, nothing sent
  python3 -m bot.live --once                # one full cycle incl. execution, then exit
  python3 -m bot.live                       # loop forever (see scripts/run_bot.sh)
  python3 -m bot.live --report              # equity, return, fees, open positions
  ROOSTOO_ENV=comp python3 -m bot.live      # competition account
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from bot.client import RoostooClient, RoostooError
from bot.execution import Executor, Order, plan_orders
from bot.ml.live_signal import LiveSignal
from bot.strategies.live_rp import RiskParitySignal
from bot.risk import exposure_scale
from bot.strategy_config import StrategyConfig, load_strategy

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR, STATE_DIR, TICKER_DIR = ROOT / "logs", ROOT / "state", ROOT / "data" / "ticker"
STOP_FILE, LIQUIDATE_FILE = ROOT / "STOP", ROOT / "LIQUIDATE"
CYCLE_S = 60.0
EXEC_BUDGET_S = 50.0            # executor time per cycle; the rest is marking + overhead
TICKER_SNAPSHOT_S = 300
STATE_VERSION = 2
EQUITY_HEADER = "t,equity,usd_free,usd_total,long_usd,short_usd,dd,scale,peak,n_pos\n"
log = logging.getLogger("live")


# ----------------------------------------------------------------------------- state
def migrate_state(old: dict, rebal_h: int, now_h: int) -> dict:
    """Bring any previous state schema to STATE_VERSION. Old schema (first dry run):
    {peak_equity, last_rebalance_hour: 'YYYY-MM-DDTHH', last_scale}."""
    new = {"version": STATE_VERSION, "hourly_marks": {}, "last_slot": None, "last_scale": 1.0,
           "start": None, "last_positions": {}, "last_plan": None}
    if not old:
        return new
    if isinstance(old.get("hourly_marks"), dict):
        new["hourly_marks"] = {str(int(k)): float(v) for k, v in old["hourly_marks"].items()}
    elif old.get("peak_equity") is not None:
        new["hourly_marks"][str(now_h)] = float(old["peak_equity"])
    if old.get("last_slot") is not None:
        new["last_slot"] = int(old["last_slot"])
    elif isinstance(old.get("last_rebalance_hour"), str):
        try:
            h = datetime.strptime(old["last_rebalance_hour"], "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
            new["last_slot"] = int(h.timestamp() // 3600) // rebal_h
        except ValueError:
            pass
    new["last_scale"] = float(old.get("last_scale", 1.0))
    for k in ("start", "last_positions", "last_plan"):
        if k in old:
            new[k] = old[k]
    return new


def load_state(path: Path, rebal_h: int, now_h: int) -> dict:
    if not path.exists():
        return migrate_state({}, rebal_h, now_h)
    try:
        old = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        log.error("state file unreadable (%s); starting fresh", e)
        return migrate_state({}, rebal_h, now_h)
    if old.get("version") == STATE_VERSION:
        return old
    new = migrate_state(old, rebal_h, now_h)
    log.info("migrated state %s -> v%d: %s", sorted(old), STATE_VERSION, {k: v for k, v in new.items() if k != "hourly_marks"})
    return new


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.replace(path)


# ----------------------------------------------------------------------------- snapshot
@dataclass
class Snapshot:
    equity: float
    usd_free: float
    usd_total: float
    long_usd: float
    short_usd: float
    pos: dict = field(default_factory=dict)     # pair -> signed USD exposure
    qty: dict = field(default_factory=dict)     # pair -> free spot quantity
    bid: dict = field(default_factory=dict)
    ask: dict = field(default_factory=dict)
    last: dict = field(default_factory=dict)
    tick: dict = field(default_factory=dict)    # raw ticker rows
    shorts: list = field(default_factory=list)  # raw short positions


def _f(d: dict, *keys, default=0.0) -> float:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            try:
                return float(d[k])
            except (TypeError, ValueError):
                continue
    return default


def short_exposure(s: dict, ask: dict) -> tuple[str | None, float, float, float]:
    """(pair, signed USD exposure, collateral, unrealised pnl) of a v6 short position row.
    Observed shape (test account, 2026-10-03): {ID, Pair, EntryPrice, ShortQty, Collateral,
    CurrentPrice, UnrealizedPNL, UnrealizedPNLPct, PositionValue, CreateTimestamp,
    PositionStatus}; other spellings are matched loosely."""
    pair = s.get("Pair") or s.get("pair") or s.get("Symbol")
    if pair and "/" not in pair and s.get("Coin"):
        pair = f"{s['Coin']}/USD"
    q = _f(s, "ShortQty", "Quantity", "quantity", "Amount", "amount", "Qty")
    coll = _f(s, "Collateral", "collateral", "CollateralValue", "Margin")
    entry = _f(s, "EntryPrice", "entry_price", "OpenPrice", "AvgPrice", "Price")
    px = ask.get(pair, 0.0) if pair else 0.0
    if q <= 0 and coll > 0 and entry > 0:
        q = coll / entry
    pnl = _f(s, "UnrealizedPNL", "UnrealizedPnL", "UnrealisedPnL", "Pnl", "PnL", default=q * (entry - px))
    return pair, -q * px, coll, pnl


def build_snapshot(tick: dict, bal: dict, shorts: list) -> Snapshot:
    """Mark to market. Longs at MaxBid, shorts at MinAsk. The USD wallet's Lock already
    contains the short collateral (ShortCollateral is reported alongside, observed on the
    test account), so equity = Free + Lock + long value + unrealised short pnl."""
    bid = {p: _f(t, "MaxBid") for p, t in tick.items()}
    ask = {p: _f(t, "MinAsk") for p, t in tick.items()}
    last = {p: _f(t, "LastPrice") for p, t in tick.items()}
    usd_b = bal.get("USD", {}) or {}
    usd_free = _f(usd_b, "Free")
    usd_lock = _f(usd_b, "Lock")
    wallet_coll = _f(usd_b, "ShortCollateral")
    pos, qty = {}, {}
    long_usd = 0.0
    for coin, b in bal.items():
        if coin == "USD":
            continue
        q = _f(b, "Free") + _f(b, "Lock")
        pair = f"{coin}/USD"
        if q > 0 and pair in bid:
            pos[pair] = pos.get(pair, 0.0) + q * bid[pair]
            qty[pair] = _f(b, "Free")
            long_usd += q * bid[pair]
    short_usd, short_pnl, pos_coll = 0.0, 0.0, 0.0
    for s in shorts:
        pair, expo, coll, pnl = short_exposure(s, ask)
        if pair is None or expo == 0:
            continue
        pos[pair] = pos.get(pair, 0.0) + expo
        short_usd += expo
        short_pnl += pnl
        pos_coll += coll
    short_val = short_pnl
    if pos_coll > 0 and wallet_coll <= 0 and usd_lock < pos_coll:
        short_val += pos_coll   # collateral not reflected in the wallet: count it from the positions
    usd_total = usd_free + usd_lock
    equity = usd_total + long_usd + short_val
    return Snapshot(equity, usd_free, usd_total, long_usd, short_usd, pos, qty, bid, ask, last, tick, shorts)


# ----------------------------------------------------------------------------- bot
class Bot:
    def __init__(self, client: RoostooClient, cfg: StrategyConfig, signal: LiveSignal | None,
                 executor: Executor, dry_run: bool, rules: dict | None = None):
        self.c, self.cfg, self.sig, self.ex, self.dry = client, cfg, signal, executor, dry_run
        self.env = client.env.lower()
        self.rules = rules if rules is not None else {k: v for k, v in client.exchange_info()["TradePairs"].items() if v.get("CanTrade")}
        self.crypto = sorted(p.split("/")[0] for p, v in self.rules.items() if v.get("AssetType") == "crypto")
        self.all_coins = sorted(p.split("/")[0] for p in self.rules)
        self.state_file = STATE_DIR / f"state_{self.env}.json"
        self.equity_file = LOG_DIR / f"equity_{self.env}.csv"
        self.state = load_state(self.state_file, cfg.rebal_h, int(time.time() // 3600))
        self.last_ticker_snapshot = 0.0
        self._shorts_logged = False
        self._liquidated = False
        self._fills_seen = 0
        self._rotate_legacy_equity_log()

    def _rotate_legacy_equity_log(self) -> None:
        """An equity csv written before the header existed is moved aside, not appended to."""
        try:
            if self.equity_file.exists() and self.equity_file.stat().st_size > 0:
                with open(self.equity_file) as f:
                    first = f.readline()
                if not first.startswith("t,"):
                    legacy = self.equity_file.with_name(self.equity_file.stem + ".legacy.csv")
                    self.equity_file.replace(legacy)
                    log.info("moved header-less equity log to %s", legacy.name)
        except OSError as e:
            log.warning("equity log rotation failed: %s", e)

    # -- exchange reads
    def snapshot(self) -> Snapshot:
        tick = self.c.ticker()
        bal = self.c.balance()
        try:
            shorts = self.c.short_positions()
        except RoostooError as e:
            if "no" in str(e).lower():  # "no open position" style message
                shorts = []
            else:
                raise
        if shorts and not self._shorts_logged:
            log.info("short_positions raw: %s", json.dumps(shorts)[:800])
            self._shorts_logged = True
        return build_snapshot(tick, bal, shorts)

    # -- logs
    def log_equity(self, now: datetime, s: Snapshot, dd: float, scale: float, peak: float) -> None:
        LOG_DIR.mkdir(exist_ok=True)
        new = not self.equity_file.exists() or self.equity_file.stat().st_size == 0
        with open(self.equity_file, "a") as f:
            if new:
                f.write(EQUITY_HEADER)
            f.write(f"{now.isoformat(timespec='seconds')},{s.equity:.2f},{s.usd_free:.2f},{s.usd_total:.2f},"
                    f"{s.long_usd:.2f},{s.short_usd:.2f},{dd:.4f},{scale:.2f},{peak:.2f},{len(s.pos)}\n")

    def log_ticker(self, now: datetime, tick: dict) -> None:
        if time.time() - self.last_ticker_snapshot < TICKER_SNAPSHOT_S:
            return
        self.last_ticker_snapshot = time.time()
        TICKER_DIR.mkdir(parents=True, exist_ok=True)
        f = TICKER_DIR / f"{now:%Y-%m-%d}.csv"
        new = not f.exists()
        ts = now.isoformat(timespec="seconds")
        with open(f, "a") as fh:
            if new:
                fh.write("t,pair,bid,ask,last,change\n")
            for pair in sorted(tick):
                t = tick[pair]
                fh.write(f"{ts},{pair},{t.get('MaxBid')},{t.get('MinAsk')},{t.get('LastPrice')},{t.get('Change')}\n")

    # -- planning
    def targets_usd(self, s: Snapshot, scale: float) -> tuple[dict, dict]:
        coins = self.crypto if self.cfg.universe == "crypto" else self.all_coins
        coins = [c for c in coins if f"{c}/USD" in s.tick]   # exchangeInfo lists pairs the ticker no longer quotes
        w, diag = self.sig.target_weights(coins)
        log.info("signal %s", json.dumps(diag, default=str))
        tgt = {}
        for pair, wt in w.items():
            if pair in self.rules and wt != 0.0:
                tgt[pair] = float(wt) * scale * s.equity
        return tgt, diag

    FORCE_AFTER_S = 6 * 3600    # activity guarantee: no fill for this long -> trade every difference

    def note_fills(self, now_ts: float | None = None) -> None:
        """Record the time of the latest filled order (drives the daily-activity guarantee)."""
        if any(str(d.get("outcome", "")).startswith("filled") or d.get("outcome") in ("short_opened", "short_closed")
               for d in self.ex.done[self._fills_seen:]):
            self.state["last_fill_ts"] = float(now_ts if now_ts is not None else time.time())
        self._fills_seen = len(self.ex.done)

    def activity_order(self, tgt: dict, s: Snapshot, prefer_sell: bool = False) -> Order | None:
        """One minimum-size order on the pair furthest from its target, in the direction of
        the target (so it never moves the book away from the strategy's weights). With
        `prefer_sell` (no USD to fund a buy) it sells from the most overweight holding."""
        best, gap = None, 0.0
        for pair in set(tgt) | set(s.pos):
            if pair not in self.rules or not s.last.get(pair):
                continue
            g = float(tgt.get(pair, 0.0)) - float(s.pos.get(pair, 0.0))
            if prefer_sell:
                if s.qty.get(pair, 0.0) > 0 and (best is None or g < gap):
                    best, gap = pair, min(g, -1e-9)
            elif abs(g) >= abs(gap):
                best, gap = pair, g
        if best is None:
            return None
        rule, px = self.rules[best], float(s.last[best])
        usd = max(self.cfg.execution.min_trade_usd, float(rule.get("MiniOrder", 0) or 0)) * 1.2
        f = 10 ** int(rule.get("AmountPrecision", 6))
        if gap >= 0:
            qty = math.ceil(usd / px * f) / f
            return Order("buy", best, qty=qty, usd=qty * px, reason="activity guarantee")
        qty = min(math.ceil(usd / px * f) / f, math.floor(float(s.qty.get(best, 0.0)) * f) / f)
        if qty * px < self.cfg.execution.min_trade_usd:
            return None
        return Order("sell", best, qty=qty, usd=qty * px, reason="activity guarantee")

    def plan(self, s: Snapshot, scale: float, liquidate: bool = False, now_ts: float | None = None) -> list[Order]:
        if liquidate:
            tgt, diag = {}, {"liquidate": True}
        else:
            tgt, diag = self.targets_usd(s, scale)
        band_usd = self.cfg.band * s.equity
        rp = self.cfg.risk_parity if self.cfg.kind == "risk_parity" else None
        if rp is not None and rp.force_daily_trade:
            now_ts = now_ts if now_ts is not None else time.time()
            if now_ts - float(self.state.get("last_fill_ts", 0.0)) > self.FORCE_AFTER_S:
                band_usd = 0.0      # every difference above the minimum order is traded
                diag = {**diag, "activity_guarantee": True}
                log.info("activity guarantee: no fill for > %dh, trading every difference", self.FORCE_AFTER_S // 3600)
        orders = plan_orders(tgt, s.pos, s.qty, s.last, self.rules, band_usd, self.cfg.execution.min_trade_usd)
        if diag.get("activity_guarantee") and not liquidate:
            min_usd = self.cfg.execution.min_trade_usd
            unfunded = all(o.kind == "buy" for o in orders) and s.usd_free < 2 * min_usd
            if not orders or unfunded:
                o = self.activity_order(tgt, s, prefer_sell=unfunded)
                if o is not None:
                    orders = [o] + orders
                    log.info("activity guarantee: minimum order %s %s %.2f USD", o.kind, o.pair, o.usd)
        log.info("targets (scale %.2f): %s", scale, {k: round(v) for k, v in tgt.items()})
        log.info("planned %d orders: %s", len(orders), [f"{o.kind} {o.pair} {round(o.usd)}" for o in orders])
        self.state["last_plan"] = {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "scale": scale,
                                   "targets_usd": {k: round(v, 2) for k, v in tgt.items()},
                                   "orders": [o.to_dict() for o in orders], "diag": diag}
        return orders

    # -- one cycle
    def step(self, force_plan: bool = False, exec_until: float | None = None) -> Snapshot:
        t0 = time.time()
        s = self.snapshot()
        now = datetime.now(timezone.utc)
        now_h = int(t0 // 3600)
        marks = {int(k): float(v) for k, v in self.state.get("hourly_marks", {}).items()}
        marks[now_h] = max(marks.get(now_h, 0.0), s.equity)
        marks = {h: v for h, v in marks.items() if h > now_h - self.cfg.risk.window_h}
        self.state["hourly_marks"] = {str(h): v for h, v in sorted(marks.items())}
        peak = max(max(marks.values()), s.equity)
        dd = s.equity / peak - 1
        scale = exposure_scale(dd, self.cfg.risk)
        if not self.state.get("start"):
            self.state["start"] = {"t": now.isoformat(timespec="seconds"), "equity": s.equity}
        self.log_equity(now, s, dd, scale, peak)
        self.log_ticker(now, s.tick)
        self.state["last_positions"] = {k: round(v, 2) for k, v in s.pos.items()}
        self.state["last_equity"] = {"t": now.isoformat(timespec="seconds"), "equity": round(s.equity, 2),
                                     "usd_free": round(s.usd_free, 2), "dd": round(dd, 4), "scale": scale}

        slot = now_h // self.cfg.rebal_h
        due = self.state.get("last_slot") != slot and now.minute >= 1   # the hourly bar has closed
        tighter = scale < float(self.state.get("last_scale", 1.0)) - 0.1
        stop, liq = STOP_FILE.exists(), LIQUIDATE_FILE.exists()
        if liq:
            if not self._liquidated and (s.pos or self.ex.busy):
                log.warning("LIQUIDATE file present: closing every position")
                self.ex.set_plan(self.plan(s, scale, liquidate=True))
                self._liquidated = True
        elif stop:
            if self.ex.busy:
                log.warning("STOP file present: cancelling queued orders")
                self.ex.cancel_all("STOP")
        else:
            self._liquidated = False
            if due or tighter or force_plan:
                why = "forced" if force_plan else ("throttle tightened" if tighter and not due else "scheduled")
                log.info("rebalance (%s): slot %s scale %.2f", why, slot, scale)
                try:
                    self.ex.set_plan(self.plan(s, scale))
                    self.state["last_slot"], self.state["last_scale"] = slot, scale
                except Exception:  # noqa: BLE001 - signal failure must not kill marking; retry next cycle
                    log.exception("planning failed; will retry next cycle")
        save_state(self.state_file, self.state)
        log.info("equity %.2f usd_free %.2f long %.0f short %.0f dd %.2f%% scale %.2f pos %d | %s%s%s",
                 s.equity, s.usd_free, s.long_usd, s.short_usd, dd * 100, scale, len(s.pos), self.ex.status(),
                 " [STOP]" if stop else "", " [LIQUIDATE]" if liq else "")
        self.ex.tick(exec_until if exec_until is not None else t0 + EXEC_BUDGET_S)
        self.note_fills()
        save_state(self.state_file, self.state)
        return s

    def run(self) -> None:
        while True:
            t0 = time.time()
            try:
                self.step()
            except Exception:  # noqa: BLE001 - keep the bot alive; errors are logged
                log.exception("step failed")
            time.sleep(max(2.0, CYCLE_S - (time.time() - t0)))


# ----------------------------------------------------------------------------- report
def report(env: str) -> str:
    """Summarise logs/equity_<env>.csv, logs/trades_<env>.jsonl and state/state_<env>.json."""
    import csv

    lines = [f"== Roostoo bot report ({env}) =="]
    eq_path, tr_path, st_path = LOG_DIR / f"equity_{env}.csv", LOG_DIR / f"trades_{env}.jsonl", STATE_DIR / f"state_{env}.json"
    state = json.loads(st_path.read_text()) if st_path.exists() else {}
    rows = []
    if eq_path.exists():
        with open(eq_path) as f:
            for r in csv.DictReader(f):
                if r.get("equity"):
                    rows.append(r)
    if rows:
        first, last = rows[0], rows[-1]
        start_eq = float((state.get("start") or {}).get("equity") or first["equity"])
        eq = float(last["equity"])
        peak, max_dd, run_peak = max(float(r["equity"]) for r in rows), 0.0, 0.0
        for r in rows:  # running-peak drawdown over the logged marks
            e = float(r["equity"])
            run_peak = max(run_peak, e)
            max_dd = min(max_dd, e / run_peak - 1)
        lines += [f"equity          {eq:,.2f} USD  (at {last['t']})",
                  f"start           {start_eq:,.2f} USD  (at {(state.get('start') or {}).get('t', first['t'])})",
                  f"return          {(eq / start_eq - 1) * 100:+.2f}%",
                  f"peak / max dd   {peak:,.2f} / {max_dd * 100:.2f}% (over logged marks)",
                  f"free USD        {float(last['usd_free']):,.2f}   long {float(last.get('long_usd', 0)):,.0f}   short {float(last.get('short_usd', 0)):,.0f}",
                  f"throttle        dd {float(last['dd']) * 100:.2f}%  scale {last['scale']}",
                  f"marks logged    {len(rows)}"]
    else:
        lines.append("no equity log yet")
    # realised fees: latest known status per OrderID from journaled query_order / fills
    orders: dict[int, dict] = {}
    legs = {"filled_limit": 0, "filled_market": 0, "filled_dry": 0, "skipped": 0, "short_opened": 0, "short_closed": 0}
    short_fees, short_pnl = 0.0, 0.0
    if tr_path.exists():
        with open(tr_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("event") == "leg_done":
                    legs[rec.get("outcome")] = legs.get(rec.get("outcome"), 0) + 1
                resp = rec.get("response")
                if rec.get("event") in ("short_open", "short_close") and isinstance(resp, dict):
                    short_fees += float(resp.get("OpenFee") or resp.get("CloseFee") or 0.0)
                    short_pnl += float(resp.get("RealizedPNL") or 0.0)
                for row in (resp or {}).get("OrderMatched", []) if isinstance(resp, dict) else []:
                    if isinstance(row, dict) and row.get("OrderID") is not None:
                        orders[int(row["OrderID"])] = row
    fees_usd, maker, taker, notional = 0.0, 0, 0, 0.0
    for row in orders.values():
        if str(row.get("Status", "")).upper() != "FILLED":
            continue
        fee = float(row.get("CommissionChargeValue") or 0.0)
        if row.get("CommissionCoin") not in ("USD", None):
            fee *= float(row.get("FilledAverPrice") or 0.0)
        fees_usd += fee
        notional += float(row.get("UnitChange") or 0.0)
        if row.get("Role") == "MAKER":
            maker += 1
        else:
            taker += 1
    lines += [f"filled orders   {maker + taker} (maker {maker}, taker {taker}), traded notional {notional:,.0f} USD",
              f"realised fees   {fees_usd + short_fees:,.2f} USD (spot {fees_usd:,.2f}, short open/close {short_fees:,.2f}); realised short pnl {short_pnl:,.2f}",
              f"legs            {', '.join(f'{k} {v}' for k, v in legs.items() if v)}" if any(legs.values()) else "legs            none"]
    pos = state.get("last_positions") or {}
    lines.append("open positions  " + (", ".join(f"{p} {v:+,.0f}" for p, v in sorted(pos.items(), key=lambda kv: -abs(kv[1]))) if pos else "none"))
    lp = state.get("last_plan") or {}
    if lp:
        lines.append(f"last plan       {lp.get('t')} scale {lp.get('scale')} targets {lp.get('targets_usd')}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="strategy JSON (default config/strategy.json)")
    ap.add_argument("--dry-run", action="store_true", help="plan and log, send nothing")
    ap.add_argument("--once", action="store_true", help="one cycle with a forced rebalance, then exit")
    ap.add_argument("--once-exec-s", type=float, default=1500.0, help="max execution time for --once (s)")
    ap.add_argument("--report", action="store_true", help="print equity / return / fees / positions from the logs")
    args = ap.parse_args()
    LOG_DIR.mkdir(exist_ok=True)
    if args.report:
        from bot.client import load_env

        load_env()
        print(report((os.environ.get("ROOSTOO_ENV") or "test").lower()))
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / "live.log")])
    cfg = load_strategy(args.config)
    client = RoostooClient()
    client.sync_time()
    log.info("starting on %s account, dry_run=%s, strategy=%s (%s)", client.env, args.dry_run, cfg.name,
             args.config or "config/strategy.json")
    rules = {k: v for k, v in client.exchange_info()["TradePairs"].items() if v.get("CanTrade")}
    executor = Executor(client, rules, client.env, cfg.execution, dry_run=args.dry_run)
    signal = RiskParitySignal(cfg.risk_parity) if cfg.kind == "risk_parity" else LiveSignal(cfg)
    bot = Bot(client, cfg, signal, executor, args.dry_run, rules=rules)
    if args.once:
        bot.step(force_plan=True, exec_until=time.time() + args.once_exec_s)
        log.info("once: done (%s)", executor.status())
    else:
        bot.run()


if __name__ == "__main__":
    main()
