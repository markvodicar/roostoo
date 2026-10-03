"""Weight functions for the core + tilt family, built from pre-computed full-panel
indicators so that EMAs / momentum have proper warm-up at a split boundary.

Everything here is picklable (plain arrays + a small callable class) so the grid
in compare_tilts.py can run in a process pool.

Portfolio = core (budget core_frac) + tilt (budget 1 - core_frac):
  core   BTC/ETH (optionally + SOL + top-3 alts by 30d dollar volume), each member
         long only while log-price > EMA(ema_span); equal weight among members that
         are "on", vol-targeted to core_vol (core_vol >= 1 => no vol targeting,
         gross capped at 1). short_core: when BTC < EMA the whole core budget
         shorts BTC (1x) instead of sitting in cash.
  tilt   "none"   cash
         "cnn"    weights_from_scores on the CNN scores of the alt universe
         "xs_mom" weights_from_scores on vol-normalised L-hour momentum of alts
         "trend"  alts with lc > EMA20d and > EMA50d; the top_k most liquid
                  (30d dollar volume) equal weight, each capped at `cap` of the budget
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from bot.ml.features import DATA
from bot.ml.portfolio import HPY, PortParams, weights_from_scores

CORE = ["BTC", "ETH"]
EXT_STATIC = ["BTC", "ETH", "SOL"]
EMA_SPANS_D = (10, 20, 30, 50)
MOM_LOOKBACKS_H = (168, 336, 720)
CORE_CORR = 0.8   # assumed pairwise correlation among core members for vol targeting


@dataclass(frozen=True)
class TiltConfig:
    family: str = "core"          # core | cnn | xs_mom | trend | ext_core | short_core
    core_frac: float = 1.0
    ema_span_d: int = 20
    core_vol: float = 0.5
    rebal_h: int = 48
    band: float = 0.02
    short_core: bool = False
    ext_core: bool = False        # core = BTC/ETH/SOL + top-3 alts by 30d dollar volume
    tilt: str = "none"            # none | cnn | xs_mom | trend
    head: int = 24                # cnn: forecast horizon
    top_k: int = 3                # cnn / xs_mom: names in the long book; trend: max names
    mom_h: int = 336              # xs_mom lookback (hours)
    tilt_vol: float = 0.5         # cnn / xs_mom: target vol of the tilt book before budget scaling
    cap: float = 0.35             # trend: max weight per name as a fraction of the tilt budget
    pred_tag: str = "ens"         # cnn: prediction tag

    @property
    def name(self) -> str:
        core = f"{'ext' if self.ext_core else 'BTC/ETH'}{'(short)' if self.short_core else ''}"
        s = f"{core} ema{self.ema_span_d}d cv{self.core_vol:g} rb{self.rebal_h} b{self.band:g} cf{self.core_frac:g}"
        if self.tilt == "cnn":
            s += f" | cnn:{self.pred_tag} h{self.head} k{self.top_k}"
        elif self.tilt == "xs_mom":
            s += f" | xsmom {self.mom_h // 24}d k{self.top_k}"
        elif self.tilt == "trend":
            s += f" | trend k{self.top_k} cap{self.cap:g}"
        return s

    def port_params(self) -> PortParams:
        return PortParams(top_k=self.top_k, head=self.head, rebal_h=self.rebal_h, band=self.band,
                          target_vol=self.tilt_vol, thresh=0.0)

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------- indicators
def load_dollar_volume(times: pd.DatetimeIndex, coins: list[str]) -> np.ndarray:
    """[T, N] hourly USD volume aligned to the panel (0 where missing)."""
    out = np.zeros((len(times), len(coins)), np.float32)
    for j, c in enumerate(coins):
        f = DATA / f"{c}.csv"
        if not f.exists():
            continue
        v = pd.read_csv(f, index_col=0, parse_dates=True, usecols=[0, 5]).iloc[:, 0]
        v = v[~v.index.duplicated()].sort_index()
        if v.index.tz is None:
            v.index = v.index.tz_localize("UTC")
        out[:, j] = v.reindex(times).fillna(0.0).values
    return out


def build_indicators(lc: np.ndarray, vol: np.ndarray, valid: np.ndarray, times: pd.DatetimeIndex,
                     coins: list[str]) -> dict:
    """Full-panel indicator arrays (no look-ahead: every value at t uses data <= t).
    lc is forward-filled so a delisted asset shows a flat price (its valid flag is False)."""
    lcf = pd.DataFrame(lc, index=times, columns=coins).ffill()
    volf = pd.DataFrame(vol, index=times, columns=coins)
    ind = {"lc": lcf.values, "vol": volf.values, "valid": valid, "coins": list(coins)}
    for d in EMA_SPANS_D:
        ema = lcf.ewm(span=24 * d, adjust=False).mean()
        ind[f"above{d}"] = (lcf > ema).values & valid
    for L in MOM_LOOKBACKS_H:
        mom = (lcf - lcf.shift(L)) / (volf * np.sqrt(L))
        ind[f"mom{L}"] = mom.where(valid).values.astype(np.float32)
    dv = pd.DataFrame(load_dollar_volume(times, coins), index=times, columns=coins)
    ind["dv30"] = dv.rolling(24 * 30, min_periods=24 * 7).mean().where(valid).values.astype(np.float32)
    return ind


def slice_indicators(ind: dict, rows: np.ndarray) -> dict:
    return {k: (v[rows] if isinstance(v, np.ndarray) and v.ndim == 2 else v) for k, v in ind.items()}


# ----------------------------------------------------------------------------- weight function
class CoreTiltWeights:
    """weight_fn(scores, vol, p, i) for bot.ml.portfolio.simulate. `ind` must already be
    sliced to the split rows (same index as the simulated frames)."""

    def __init__(self, cfg: TiltConfig, ind: dict):
        self.cfg = cfg
        self.coins = ind["coins"]
        self.pos = {c: k for k, c in enumerate(self.coins)}
        self.above = ind[f"above{cfg.ema_span_d}"]
        self.valid = ind["valid"]
        self.vol = ind["vol"]
        self.static = EXT_STATIC if cfg.ext_core else CORE
        self.static_idx = np.array([self.pos[c] for c in self.static])
        self.dv30 = ind["dv30"] if cfg.ext_core or cfg.tilt == "trend" else None
        self.mom = ind[f"mom{cfg.mom_h}"] if cfg.tilt == "xs_mom" else None
        if cfg.tilt == "trend":
            self.tr_ok = ind["above20"] & ind["above50"]
        self.btc = self.pos["BTC"]

    def core_members(self, i: int) -> np.ndarray:
        if not self.cfg.ext_core:
            return self.static_idx
        dv = self.dv30[i].copy()
        dv[self.static_idx] = np.nan
        dv[~self.valid[i]] = np.nan
        extra = np.argsort(-np.nan_to_num(dv, nan=-1.0))[:3]
        extra = extra[np.isfinite(dv[extra])]
        return np.concatenate([self.static_idx, extra])

    def core_weights(self, i: int) -> np.ndarray:
        cfg = self.cfg
        w = np.zeros(len(self.coins))
        members = self.core_members(i)
        hv = np.nan_to_num(self.vol[i], nan=1e-4)
        if cfg.short_core and not self.above[i, self.btc]:
            cv = hv[self.btc] * np.sqrt(HPY)
            w[self.btc] = -min(cfg.core_vol / max(cv, 1e-9), 1.0) * cfg.core_frac
            return w
        on = self.above[i, members]
        if on.sum() == 0:
            return w
        cw = on / on.sum()
        cv = hv[members] * np.sqrt(HPY)
        x = cw * cv
        pv = float(np.sqrt((x ** 2).sum() + CORE_CORR * (x.sum() ** 2 - (x ** 2).sum())))
        w[members] = cw * min(cfg.core_vol / max(pv, 1e-9), 1.0) * cfg.core_frac
        return w

    def tilt_weights(self, scores: pd.Series, vol: pd.Series, p: PortParams, i: int, members: np.ndarray) -> pd.Series:
        cfg = self.cfg
        budget = 1.0 - cfg.core_frac
        excl = [self.coins[k] for k in members]
        if cfg.tilt == "cnn":
            return weights_from_scores(scores.drop(excl, errors="ignore"), vol, p) * budget
        if cfg.tilt == "xs_mom":
            s = pd.Series(self.mom[i], index=self.coins).drop(excl)
            return weights_from_scores(s, vol, p) * budget
        if cfg.tilt == "trend":
            ok = self.tr_ok[i].copy()
            ok[members] = False
            dv = np.where(ok, np.nan_to_num(self.dv30[i], nan=0.0), -1.0)
            pick = np.argsort(-dv)[: cfg.top_k]
            pick = pick[dv[pick] > 0]
            w = pd.Series(0.0, index=self.coins)
            if len(pick):
                w.iloc[pick] = min(1.0 / len(pick), cfg.cap) * budget
            return w
        return pd.Series(0.0, index=self.coins)

    def __call__(self, scores: pd.Series, vol: pd.Series, p: PortParams, i: int) -> pd.Series:
        members = self.core_members(i)
        w = pd.Series(self.core_weights(i), index=self.coins)
        if self.cfg.core_frac < 1.0 and self.cfg.tilt != "none":
            w = w.add(self.tilt_weights(scores, vol, p, i, members), fill_value=0.0)
        g = w.abs().sum()
        if g > 1.0:
            w /= g
        return w
