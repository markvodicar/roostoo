"""Thin, defensive client for the Roostoo mock exchange API.

Signing: params sorted by key, joined as k=v&k=v, HMAC-SHA256 with the secret,
hex digest sent in MSG-SIGNATURE alongside RST-API-KEY. All endpoints return
HTTP 200 with Success=false + ErrMsg on logical failure.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from pathlib import Path

import requests

BASE_URL = "https://mock-api.roostoo.com"
log = logging.getLogger(__name__)


def load_env(path: str | Path = Path(__file__).resolve().parent.parent / ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


class RoostooError(RuntimeError):
    pass


class RoostooClient:
    def __init__(self, api_key: str | None = None, secret: str | None = None,
                 env: str | None = None, timeout: float = 10.0, retries: int = 3):
        load_env()
        env = (env or os.environ.get("ROOSTOO_ENV", "test")).upper()
        self.env = env
        self.api_key = api_key or os.environ[f"ROOSTOO_{env}_API_KEY"]
        self.secret = (secret or os.environ[f"ROOSTOO_{env}_API_SECRET"]).encode()
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self._time_offset_ms = 0

    # ---------- plumbing ----------
    def _ts(self) -> str:
        return str(int(time.time() * 1000) + self._time_offset_ms)

    def _sign(self, params: dict) -> str:
        qs = "&".join(f"{k}={params[k]}" for k in sorted(params))
        return hmac.new(self.secret, qs.encode(), hashlib.sha256).hexdigest()

    def _request(self, method: str, path: str, params: dict | None = None,
                 signed: bool = False, ts: bool = False) -> dict:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            p = dict(params)
            if ts or signed:
                p["timestamp"] = self._ts()
            headers = {}
            if signed:
                headers["RST-API-KEY"] = self.api_key
                headers["MSG-SIGNATURE"] = self._sign(p)
            try:
                if method == "GET":
                    r = self.session.get(BASE_URL + path, params=p, headers=headers, timeout=self.timeout)
                else:
                    headers["Content-Type"] = "application/x-www-form-urlencoded"
                    r = self.session.post(BASE_URL + path, data=p, headers=headers, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
            except (requests.RequestException, ValueError) as e:
                last_exc = e
                log.warning("%s %s failed (attempt %d): %s", method, path, attempt + 1, e)
                time.sleep(1.5 * (attempt + 1))
                continue
            if isinstance(data, dict) and data.get("Success") is False:
                msg = data.get("ErrMsg", "")
                # Clock drift: resync once and retry.
                if "timestamp" in msg.lower() and attempt < self.retries - 1:
                    self.sync_time()
                    continue
                raise RoostooError(f"{path}: {msg}")
            return data
        raise RoostooError(f"{path}: giving up after {self.retries} attempts: {last_exc}")

    # ---------- public ----------
    def server_time(self) -> int:
        return int(self._request("GET", "/v3/serverTime")["ServerTime"])

    def sync_time(self) -> None:
        t0 = time.time() * 1000
        st = self.server_time()
        t1 = time.time() * 1000
        self._time_offset_ms = int(st - (t0 + t1) / 2)

    def exchange_info(self) -> dict:
        return self._request("GET", "/v3/exchangeInfo")

    def ticker(self, pair: str | None = None) -> dict:
        return self._request("GET", "/v3/ticker", {"pair": pair}, ts=True)["Data"]

    # ---------- signed ----------
    def balance(self) -> dict:
        d = self._request("GET", "/v3/balance", signed=True)
        return d.get("SpotWallet") or d.get("Wallet") or {}

    def pending_count(self) -> dict:
        return self._request("GET", "/v3/pending_count", signed=True)

    def place_order(self, pair: str, side: str, quantity, order_type: str = "MARKET", price=None) -> dict:
        params = {"pair": pair, "side": side.upper(), "type": order_type.upper(), "quantity": quantity}
        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("LIMIT order needs price")
            params["price"] = price
        return self._request("POST", "/v3/place_order", params, signed=True)

    def query_order(self, order_id=None, pair=None, pending_only=None, offset=None, limit=None) -> dict:
        po = None if pending_only is None else ("TRUE" if pending_only else "FALSE")
        return self._request("POST", "/v3/query_order",
                             {"order_id": order_id, "pair": pair, "pending_only": po,
                              "offset": offset, "limit": limit}, signed=True)

    def cancel_order(self, order_id=None, pair=None) -> dict:
        return self._request("POST", "/v3/cancel_order", {"order_id": order_id, "pair": pair}, signed=True)

    # ---------- shorts (v6; collateral in USD, qty = collateral / entry, 0.1% open + 0.1% close) ----------
    def short_open(self, pair: str, collateral_usd: float) -> dict:
        return self._request("POST", "/v6/short_open", {"pair": pair, "collateral": collateral_usd}, signed=True)

    def short_close(self, pair: str, close_pct: float | None = None) -> dict:
        return self._request("POST", "/v6/short_close", {"pair": pair, "close_pct": close_pct}, signed=True)

    def short_positions(self) -> list[dict]:
        d = self._request("GET", "/v6/short_positions", signed=True)
        for key in ("Data", "Positions", "positions"):
            if isinstance(d.get(key), list):
                return d[key]
        return [v for v in d.values() if isinstance(v, list)][0] if any(isinstance(v, list) for v in d.values()) else []
