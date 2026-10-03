"""Full 2016->today hourly history for model training.

* Binance spot USDT klines from each coin's listing (Binance opened Jul 2017).
* Coinbase USD candles for BTC/ETH/LTC fill 2016 -> Binance start (spliced: Binance
  wins wherever both exist; USD vs USDT basis is negligible at 1h horizons).
Saved as data/full/<COIN>.csv with open, high, low, close, volume_usd.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parent.parent / "data" / "full"
START = datetime(2016, 1, 1, tzinfo=timezone.utc)
COINBASE_FILL = {"BTC": "BTC-USD", "ETH": "ETH-USD", "LTC": "LTC-USD"}


def _get(url, params, tries=6):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 * (i + 1))
                continue
            return r
        except requests.RequestException:
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"failed {url} {params}")


def binance(coin: str) -> pd.DataFrame:
    sym, start, end, rows = coin + "USDT", int(START.timestamp() * 1000), int(time.time() * 1000), []
    while start < end:
        r = _get("https://api.binance.com/api/v3/klines",
                 {"symbol": sym, "interval": "1h", "startTime": start, "limit": 1000})
        if r.status_code == 400:
            return pd.DataFrame()
        b = r.json()
        if not b:
            break
        rows += b
        start = b[-1][0] + 3_600_000
        if len(b) < 1000:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([[x[0], x[1], x[2], x[3], x[4], x[7]] for x in rows],
                      columns=["t", "open", "high", "low", "close", "volume_usd"])
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    return df.set_index("t").astype(float)


def coinbase(product: str, until: pd.Timestamp) -> pd.DataFrame:
    rows, t = [], START
    step = pd.Timedelta(hours=300)
    t = pd.Timestamp(START)
    while t < until:
        e = min(t + step, until)
        r = _get(f"https://api.exchange.coinbase.com/products/{product}/candles",
                 {"granularity": 3600, "start": t.isoformat(), "end": e.isoformat()})
        if r.status_code == 200:
            rows += r.json()
        t = e
        time.sleep(0.15)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["t", "low", "high", "open", "close", "volume"])
    df["t"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df = df.drop_duplicates("t").set_index("t").sort_index().astype(float)
    df["volume_usd"] = df["volume"] * df["close"]
    return df[["open", "high", "low", "close", "volume_usd"]]


def build(coin: str) -> str:
    f = OUT / f"{coin}.csv"
    if f.exists():
        return f"{coin}: cached"
    df = binance(coin)
    if coin in COINBASE_FILL:
        until = df.index[0] if not df.empty else pd.Timestamp.now(tz="UTC")
        cb = coinbase(COINBASE_FILL[coin], until)
        df = pd.concat([cb[cb.index < until], df]) if not df.empty else cb
    if df.empty:
        return f"{coin}: no data"
    df = df[~df.index.duplicated()].sort_index()
    df.to_csv(f)
    return f"{coin}: {len(df)} bars {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    info = requests.get("https://mock-api.roostoo.com/v3/exchangeInfo", timeout=10).json()
    coins = sorted(k.split("/")[0] for k, v in info["TradePairs"].items() if v.get("AssetType") == "crypto")
    with ThreadPoolExecutor(6) as ex:
        for msg in ex.map(build, coins):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
