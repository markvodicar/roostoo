"""Historical data for backtesting.

Roostoo only exposes a live ticker, so history comes from Binance public klines
(Roostoo crypto prices track Binance spot USDT pairs). Cached as CSV in data/.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BINANCE = "https://api.binance.com/api/v3/klines"


def roostoo_to_binance(pair: str) -> str:
    return pair.split("/")[0] + "USDT"


def fetch_klines(symbol: str, interval: str = "1h", days: int = 365) -> pd.DataFrame:
    end = int(time.time() * 1000)
    start = end - days * 86_400_000
    rows = []
    while start < end:
        r = requests.get(BINANCE, params={"symbol": symbol, "interval": interval,
                                          "startTime": start, "limit": 1000}, timeout=15)
        if r.status_code == 400:  # symbol not listed on Binance
            return pd.DataFrame()
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows += batch
        start = batch[-1][0] + 1
        if len(batch) < 1000:
            break
        time.sleep(0.05)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume", "ct", "qv",
                                     "n", "tb", "tq", "_"])
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.set_index("t")[["open", "high", "low", "close", "volume", "qv"]].astype(float)
    return df[~df.index.duplicated()]


def download_universe(pairs: list[str], interval: str = "1h", days: int = 365, refresh: bool = False) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for pair in pairs:
        sym = roostoo_to_binance(pair)
        f = DATA_DIR / f"{sym}_{interval}.csv"
        if f.exists() and not refresh:
            continue
        df = fetch_klines(sym, interval, days)
        if df.empty:
            print(f"  {pair}: not on Binance, skipped")
            continue
        df.to_csv(f)
        print(f"  {pair}: {len(df)} bars from {df.index[0]:%Y-%m-%d}")


def load_panel(pairs: list[str], interval: str = "1h", field: str = "close") -> pd.DataFrame:
    cols = {}
    for pair in pairs:
        f = DATA_DIR / f"{roostoo_to_binance(pair)}_{interval}.csv"
        if f.exists():
            cols[pair] = pd.read_csv(f, index_col=0, parse_dates=True)[field]
    return pd.DataFrame(cols).sort_index()


if __name__ == "__main__":
    import sys

    info = requests.get("https://mock-api.roostoo.com/v3/exchangeInfo", timeout=10).json()
    pairs = sorted(p for p, v in info["TradePairs"].items() if v.get("AssetType") == "crypto")
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 365
    print(f"Downloading {len(pairs)} crypto pairs, {days}d of 1h bars")
    download_universe(pairs, "1h", days)
