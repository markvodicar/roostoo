"""Live inference: fresh hourly bars -> core + tilt target weights.

The maths is the backtest's, re-used rather than re-implemented where possible:
  * features / vol / log close come from bot.ml.features.build_panel
  * the tilt sizing is bot.ml.portfolio.weights_from_scores
  * the core sleeve (EMA gate, vol target with core_corr correlation, optional BTC
    short overlay, tilt budget 1 - core_frac, gross cap 1) is `core_tilt_weights`, a
    line-for-line port of bot.research.tilts.CoreTiltWeights (and of
    bot.ml.coretilt.make_weight_fn for the long core) evaluated at the last row;
    tests/test_weights.py and tests/test_short_overlay.py assert equality.

Tilt variants (StrategyConfig.tilt):
  cnn     rank-averaged CNN score at horizon `head` among alts, weights_from_scores
  xs_mom  cross-sectional vol-normalised momentum: mean over 7/14/30d of
          log-return / (vol * sqrt(h)), same sizing as the CNN tilt
  trend   price above EMA(20d) and EMA(50d): equal weight among qualifying alts,
          each capped at max_w, scaled by the tilt budget
  none    core only

Bars: Binance 1h klines (Roostoo crypto prices track Binance spot), cached in
memory and on disk (data/live_bars/<COIN>.csv) and refreshed once per closed
hour, so the 60s bot loop never re-downloads. Stocks tokens have no history and
are never scored; the bot still marks and can liquidate them.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from bot.data import fetch_klines
from bot.ml.features import HORIZONS, build_panel, regularize
from bot.ml.portfolio import HPY, PortParams, weights_from_scores
from bot.strategy_config import StrategyConfig

ROOT = Path(__file__).resolve().parents[2]
BAR_DIR = ROOT / "data" / "live_bars"
MODEL_DIR = ROOT / "models"
log = logging.getLogger("signal")
HISTORY_DAYS = 120          # EMA(50d) warm-up + 96h window + MIN_BARS; >= 90 required
MOM_HORIZONS_H = (168, 336, 720)
TREND_SPANS_H = (480, 1200)


# ----------------------------------------------------------------------------- weights
def core_weights(lcf: pd.DataFrame, vol: pd.Series, core: list[str], core_frac: float,
                 ema_span: int, core_vol: float, short_core: bool = False, core_corr: float = 0.8,
                 valid: pd.Series | None = None) -> pd.Series:
    """Core sleeve at the last row of `lcf` (log close, columns >= core coins). Line-for-line
    port of bot.research.tilts.CoreTiltWeights.core_weights evaluated at one row:
      gate     log close > EMA(ema_span, adjust=False), and `valid` (default: price present)
      long     equal weight among gated coins, scaled by min(core_vol / pv, 1) * core_frac where
               pv uses hourly vol * sqrt(8760) and pairwise correlation core_corr
      short    (short_core) when BTC is NOT gated on, the whole budget is a BTC short of
               -min(core_vol / BTC_ann_vol, 1) * core_frac and nothing else is held
    NaN vol is replaced by 1e-4 as in the reference."""
    core = [c for c in core if c in lcf.columns]
    w = pd.Series(0.0, index=core, dtype=float)
    if not core or core_frac <= 0:
        return w
    ema = lcf[core].ewm(span=ema_span, adjust=False).mean()
    last = lcf[core].iloc[-1]
    ok = last.notna() if valid is None else valid.reindex(core).fillna(False).astype(bool)
    above = ((last > ema.iloc[-1]) & ok).values
    hv = np.nan_to_num(vol.reindex(core).values.astype(float), nan=1e-4)
    if short_core and "BTC" in core and not above[core.index("BTC")]:
        cv = hv[core.index("BTC")] * np.sqrt(HPY)
        w["BTC"] = -min(core_vol / max(cv, 1e-9), 1.0) * core_frac
        return w
    on = above.astype(float)
    if on.sum() == 0:
        return w
    cw = on / on.sum()
    cv = hv * np.sqrt(HPY)
    x = cw * cv
    pv = float(np.sqrt((x ** 2).sum() + core_corr * (x.sum() ** 2 - (x ** 2).sum())))
    w[:] = cw * min(core_vol / max(pv, 1e-9), 1.0) * core_frac
    return w


def core_tilt_weights(lcf: pd.DataFrame, vol: pd.Series, scores: pd.Series | None, core: list[str],
                      core_frac: float, ema_span: int, core_vol: float, p: PortParams,
                      short_core: bool = False, tilt_weights: pd.Series | None = None,
                      core_corr: float = 0.8, valid: pd.Series | None = None) -> pd.Series:
    """Full core + tilt target at the last row. `scores` (any coins; core dropped) feeds
    weights_from_scores; alternatively pass precomputed unit-budget `tilt_weights`.
    Gross exposure is capped at 1 (no leverage)."""
    w = core_weights(lcf, vol, core, core_frac, ema_span, core_vol, short_core, core_corr, valid)
    if core_frac < 1.0:
        if tilt_weights is None and scores is not None:
            tilt_weights = weights_from_scores(scores.drop(core, errors="ignore"), vol, p)
        if tilt_weights is not None:
            w = w.add(tilt_weights * (1 - core_frac), fill_value=0.0)
    g = w.abs().sum()
    if g > 1.0:
        w /= g
    return w


def xs_mom_scores(lc: np.ndarray, vol: np.ndarray, valid: np.ndarray, coins: list[str]) -> pd.Series:
    """Cross-sectional momentum at the last row: mean over MOM_HORIZONS_H of
    (lc[T] - lc[T-h]) / (vol[T] * sqrt(h)). NaN where history is missing."""
    T = len(lc) - 1
    parts = []
    for h in MOM_HORIZONS_H:
        if T - h < 0:
            return pd.Series(np.nan, index=coins)
        parts.append((lc[T] - lc[T - h]) / (vol[T] * np.sqrt(h)))
    s = pd.Series(np.nanmean(np.vstack(parts), axis=0), index=coins)
    return s.where(valid[T])


def trend_weights(lcf: pd.DataFrame, valid_last: pd.Series, exclude: list[str], max_w: float) -> pd.Series:
    """Unit-budget trend tilt: alts with close above every EMA in TREND_SPANS_H get
    min(1/n, max_w) each."""
    alts = [c for c in lcf.columns if c not in exclude and bool(valid_last.get(c, False))]
    if not alts:
        return pd.Series(dtype=float)
    last = lcf[alts].iloc[-1]
    on = pd.Series(True, index=alts)
    for span in TREND_SPANS_H:
        on &= last > lcf[alts].ewm(span=span, adjust=False).mean().iloc[-1]
    sel = on[on].index
    if len(sel) == 0:
        return pd.Series(dtype=float)
    return pd.Series(min(1.0 / len(sel), max_w), index=sel)


# ----------------------------------------------------------------------------- bars
class BarCache:
    """Hourly OHLCV per coin, refreshed at most once per closed hour.

    First call downloads `days` of history per coin (parallel); later calls fetch only
    the tail and splice it in. Frames are persisted as CSV so a restart is cheap. The
    still-forming bar (open time == current hour) is always dropped."""

    def __init__(self, days: int = HISTORY_DAYS, bar_dir: Path = BAR_DIR, fetch=fetch_klines, workers: int = 6):
        self.days, self.dir, self.fetch, self.workers = days, Path(bar_dir), fetch, workers
        self.frames: dict[str, pd.DataFrame] = {}
        self.slot = -1
        self.failed: list[str] = []

    @staticmethod
    def _hour_floor(ts: float | None = None) -> pd.Timestamp:
        return pd.Timestamp(int((ts or time.time()) // 3600 * 3600), unit="s", tz="UTC")

    def _load_disk(self, coin: str) -> pd.DataFrame | None:
        f = self.dir / f"{coin}.csv"
        if not f.exists():
            return None
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            df.index = pd.DatetimeIndex(df.index).tz_convert("UTC") if df.index.tz else pd.DatetimeIndex(df.index).tz_localize("UTC")
            return df if len(df) else None
        except Exception as e:  # noqa: BLE001 - corrupt cache file: refetch
            log.warning("bar cache %s unreadable (%s); refetching", coin, e)
            return None

    def _refresh_one(self, coin: str, now_h: pd.Timestamp) -> pd.DataFrame | None:
        old = self.frames.get(coin)
        if old is None:
            old = self._load_disk(coin)
        start_cut = now_h - pd.Timedelta(days=self.days)
        if old is not None and old.index[-1] >= start_cut and old.index[0] <= start_cut + pd.Timedelta(days=2):
            gap_days = max(1, int((now_h - old.index[-1]).total_seconds() // 86400) + 2)
            new = self.fetch(coin + "USDT", "1h", days=gap_days)
            df = pd.concat([old, new]) if new is not None and len(new) else old
        else:
            df = self.fetch(coin + "USDT", "1h", days=self.days)
        if df is None or df.empty:
            return None
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df[(df.index >= start_cut) & (df.index < now_h)]
        return df

    def get(self, coins: list[str]) -> dict[str, pd.DataFrame]:
        """Returns {coin: regularized OHLCV frame with columns open/high/low/close/volume_usd}."""
        now_h = self._hour_floor()
        slot = int(now_h.timestamp() // 3600)
        need = [c for c in coins if c not in self.frames] if slot == self.slot else list(coins)
        if need:
            self.dir.mkdir(parents=True, exist_ok=True)
            failed = []

            def job(c):
                try:
                    return c, self._refresh_one(c, now_h)
                except Exception as e:  # noqa: BLE001 - one coin's download failing must not stop the bot
                    log.warning("bars %s: %s", c, e)
                    return c, None

            with ThreadPoolExecutor(self.workers) as ex:
                for c, df in ex.map(job, need):
                    if df is None:
                        failed.append(c)
                        continue
                    self.frames[c] = df
                    try:
                        df.to_csv(self.dir / f"{c}.csv")
                    except OSError as e:
                        log.warning("bar cache write %s: %s", c, e)
            self.failed = failed
            self.slot = slot
            if failed:
                log.warning("no bars for %d coins: %s", len(failed), failed)
        out = {}
        for c in coins:
            df = self.frames.get(c)
            if df is None or len(df) < 48:
                continue
            df = df.rename(columns={"qv": "volume_usd"})
            cols = ["open", "high", "low", "close", "volume_usd"]
            if not set(cols) <= set(df.columns):
                continue
            out[c] = regularize(df[cols].copy())
        return out

    def last_bar(self) -> pd.Timestamp | None:
        ends = [df.index[-1] for df in self.frames.values() if len(df)]
        return max(ends) if ends else None


# ----------------------------------------------------------------------------- signal
class LiveSignal:
    def __init__(self, cfg: StrategyConfig, bars: BarCache | None = None):
        self.cfg, self.p = cfg, cfg.tilt_params
        self.bars = bars or BarCache()
        self.models = []
        if cfg.tilt == "cnn" and cfg.core_frac < 1.0:
            import torch  # heavy import only when the CNN tilt is active
            from bot.ml.model import CNN

            for t in cfg.models:
                ck = torch.load(MODEL_DIR / f"{t}.pt", map_location="cpu", weights_only=False)
                m = CNN(hidden=ck["hidden"], drop=ck["drop"])
                m.load_state_dict(ck["state"])
                m.eval()
                self.models.append(m)
                log.info("loaded model %s (val epoch %s, ic_336 %.4f)", t, ck.get("epoch"), ck.get("val", {}).get("ic_336", float("nan")))
            self.head = HORIZONS.index(self.p.head)
        log.info("signal: core=%s frac=%.2f ema=%dh cv=%.2f tilt=%s %s", cfg.core, cfg.core_frac,
                 cfg.ema_span_h, cfg.core_vol, cfg.tilt, self.p)

    # -- model scores at the last bar
    def cnn_scores(self, feat, valid, coins) -> pd.Series:
        import torch
        from bot.ml.model import WINDOW

        T = len(feat) - 1
        ok = valid[T] & valid[T - WINDOW + 1: T + 1].all(0) if T + 1 >= WINDOW else np.zeros(len(coins), bool)
        js = np.where(ok)[0]
        if len(js) == 0:
            return pd.Series(dtype=float)
        x = torch.from_numpy(np.stack([feat[T - WINDOW + 1: T + 1, j] for j in js]))
        names = [coins[j] for j in js]
        with torch.no_grad():
            preds = [pd.Series(m(x).numpy()[:, self.head], index=names) for m in self.models]
        if len(preds) == 1:
            return preds[0]
        # same transform as bot/ml/ensemble.py: centred mean percentile rank, rescaled
        return (sum(p.rank(pct=True) for p in preds) / len(preds) - 0.5) * 2 * pd.concat(preds).std()

    def target_weights(self, coins: list[str]) -> tuple[pd.Series, dict]:
        """Signed target weights keyed by 'COIN/USD' (gross <= 1) plus diagnostics.
        `coins` are the scorable universe; coins without bars are skipped."""
        cfg, p = self.cfg, self.p
        if cfg.tilt == "none" or cfg.core_frac >= 1.0:
            coins = [c for c in coins if c in cfg.core]   # core only: no alt bars needed
        ohlcv = self.bars.get(coins)
        if not ohlcv:
            raise RuntimeError("no bars available for any coin")
        feat, vol, lc, valid, times, pcoins = build_panel(ohlcv)
        T = len(times) - 1
        lcf = pd.DataFrame(lc, index=times, columns=pcoins).ffill()
        vol_last = pd.Series(vol[T], index=pcoins)
        valid_last = pd.Series(valid[T], index=pcoins)
        mkt = float(feat[T, 0, 7])
        core = [c for c in cfg.core if c in pcoins]
        missing_core = [c for c in cfg.core if c not in pcoins]
        if missing_core:
            log.warning("core coins without bars: %s", missing_core)

        scores: pd.Series | None = None
        tilt_w: pd.Series | None = None
        if cfg.core_frac < 1.0:
            if cfg.tilt == "cnn":
                scores = self.cnn_scores(feat, valid, pcoins)
            elif cfg.tilt == "xs_mom":
                scores = xs_mom_scores(lc, vol, valid, pcoins).dropna()
            elif cfg.tilt == "trend":
                tilt_w = trend_weights(lcf, valid_last, core, p.max_w)
            if scores is not None:
                scores = scores.drop(core, errors="ignore")
                if p.use_regime and mkt < 0:
                    scores = scores.where(scores < 0)
        w = core_tilt_weights(lcf, vol_last, scores, core, cfg.core_frac, cfg.ema_span_h, cfg.core_vol, p,
                              short_core=cfg.short_core, tilt_weights=tilt_w, core_corr=cfg.core_corr,
                              valid=valid_last)
        w.index = [c + "/USD" for c in w.index]
        ema = lcf[core].ewm(span=cfg.ema_span_h, adjust=False).mean().iloc[-1] if core else pd.Series(dtype=float)
        diag = {"bar": str(times[T]), "mkt": round(mkt, 3), "n_bars": len(ohlcv), "tilt": cfg.tilt,
                "short_core": cfg.short_core,
                "core_above_ema": {c: bool(lcf[c].iloc[-1] > ema[c]) for c in core},
                "core_valid": {c: bool(valid_last[c]) for c in core},
                "core_vol_ann": {c: round(float(vol_last[c] * np.sqrt(HPY)), 3) for c in core},
                "gross": round(float(w.abs().sum()), 3)}
        if scores is not None and len(scores.dropna()):
            s = scores.dropna()
            diag["n_scored"] = int(len(s))
            diag["top"] = {c: round(float(v), 3) for c, v in s.nlargest(5).items()}
            diag["bottom"] = {c: round(float(v), 3) for c, v in s.nsmallest(3).items()}
        return w, diag
