"""Survivorship-free daily dataset: every USDT spot pair Binance has ever listed.

* Pairs still trading: /api/v3/klines (1d).
* Delisted pairs: monthly 1d archives from data.binance.vision.
* Excluded: leveraged tokens (*UP/*DOWN/*BULL/*BEAR), fiat and stablecoins (explicit list
  plus a realised-vol screen applied at load time).
Symbol reuse (e.g. LUNA after the 2022 collapse) and redenominations are handled at load
time by splitting a series into separate assets at gaps > 20 days, or at gaps > 2 days
with a > 5x price break.

    python3 -m bot.research.pit_data          # download (idempotent, cached)
"""
from __future__ import annotations

import io
import os
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUT = os.path.join(ROOT, "data", "binance_all")
ARCH = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FILES = "https://data.binance.vision"
STABLE = {"USDC", "BUSD", "TUSD", "PAX", "USDP", "FDUSD", "DAI", "UST", "USDS", "USDSB", "SUSD",
          "EUR", "GBP", "AUD", "BRL", "TRY", "RUB", "NGN", "UAH", "ZAR", "BIDR", "IDRT", "BKRW",
          "AEUR", "EURI", "USDE", "XUSD", "USD1", "RLUSD", "BFUSD", "PAXG", "WBTC", "WBETH", "BETH",
          "BTCST", "USTC", "ERD"}
COLS = ["t", "open", "high", "low", "close", "volume", "ct", "qv", "n", "tb", "tq", "_"]


def list_usdt() -> list[str]:
    syms, marker = [], ""
    while True:
        r = requests.get(ARCH, params={"delimiter": "/", "prefix": "data/spot/monthly/klines/",
                                       "marker": marker}, timeout=30)
        pre = re.findall(r"<Prefix>data/spot/monthly/klines/([^<]+)/</Prefix>", r.text)
        syms += pre
        if "<IsTruncated>true</IsTruncated>" not in r.text:
            break
        marker = "data/spot/monthly/klines/" + pre[-1] + "/"
    out = []
    for s in syms:
        if not s.endswith("USDT"):
            continue
        base = s[:-4]
        if re.search(r"(UP|DOWN|BULL|BEAR)$", base) and len(base) > 4:
            continue
        if base in STABLE:
            continue
        out.append(s)
    return out


def _api(sym):
    rows, start = [], 1500000000000
    while True:
        r = requests.get("https://api.binance.com/api/v3/klines",
                         params={"symbol": sym, "interval": "1d", "startTime": start, "limit": 1000}, timeout=30)
        if r.status_code == 400:
            return None
        r.raise_for_status()
        b = r.json()
        if not b:
            break
        rows += b
        start = b[-1][0] + 86_400_000
        if len(b) < 1000:
            break
    return rows


def _archive(sym):
    r = requests.get(ARCH, params={"prefix": f"data/spot/monthly/klines/{sym}/1d/"}, timeout=30)
    keys = [k for k in re.findall(r"<Key>([^<]+)</Key>", r.text) if k.endswith(".zip")]
    rows = []
    for k in keys:
        for _ in range(4):
            try:
                z = requests.get(f"{FILES}/{k}", timeout=60)
                z.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
                    for name in zf.namelist():
                        for line in zf.read(name).decode().splitlines():
                            p = line.split(",")
                            if p and p[0].isdigit():
                                rows.append(p[:12])
                break
            except Exception:  # noqa: BLE001
                time.sleep(2)
    return rows


def fetch(sym):
    f = os.path.join(OUT, f"{sym}.csv")
    if os.path.exists(f):
        return f"{sym}: cached"
    try:
        rows = _api(sym)
        src = "api"
        if not rows:
            rows, src = _archive(sym), "archive"
        if not rows:
            return f"{sym}: no data"
        df = pd.DataFrame(rows, columns=COLS)
        t = pd.to_numeric(df["t"])
        t = np.where(t > 1e14, t // 1000, t)          # archives after 2025 use microseconds
        df["t"] = pd.to_datetime(t, unit="ms", utc=True)
        df = df.set_index("t")[["open", "high", "low", "close", "volume", "qv"]].astype(float)
        df = df[~df.index.duplicated()].sort_index()
        df.to_csv(f)
        return f"{sym}: {src} {len(df)} days {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}"
    except Exception as e:  # noqa: BLE001
        return f"{sym}: ERROR {e}"


def load(min_days=30, gap_days=20):
    """Return {asset_id: daily DataFrame}; series split at gaps > gap_days (symbol reuse)."""
    out = {}
    for fn in sorted(os.listdir(OUT)):
        if not fn.endswith(".csv"):
            continue
        d = pd.read_csv(os.path.join(OUT, fn), index_col=0, parse_dates=True)
        d = d[d["close"] > 0]
        if len(d) < min_days:
            continue
        sym = fn[:-8]
        gap = d.index.to_series().diff()
        jump = (d["close"] / d["close"].shift(1)).values
        # new segment after a long gap, or after any >2-day gap with a >5x price break
        # (symbol reuse such as LUNA in May 2022, or token redenominations)
        brk = (gap > pd.Timedelta(days=gap_days)) | ((gap > pd.Timedelta(days=2)) & ((jump > 5) | (jump < 0.2)))
        seg_id = brk.cumsum()
        for k, seg in d.groupby(seg_id.values):
            if len(seg) < min_days:
                continue
            r = np.log(seg["close"]).diff()
            if r.std() * np.sqrt(365) < 0.08:      # stablecoin / pegged asset
                continue
            out[sym if k == 0 else f"{sym}#{k}"] = seg
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    syms = list_usdt()
    print(f"{len(syms)} candidate USDT pairs", flush=True)
    with ThreadPoolExecutor(8) as ex:
        for i, msg in enumerate(ex.map(fetch, syms)):
            if "ERROR" in msg or "no data" in msg or i % 50 == 0:
                print(i, msg, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
