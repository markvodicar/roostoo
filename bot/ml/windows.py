"""Rolling 14-day window evaluation: the competition is scored on ONE two-week window
with unknown start, so the objective is the distribution over all daily-rolling
14-day windows, not the full-period statistics.

    from bot.ml.windows import rolling_windows, summarize

`rolling_windows` reproduces `bot.backtest.metrics` on every window segment exactly
(same calendar-day resample bins, same ddof, same annualisation) but computes it in
numpy from one daily resample of the whole curve, which is ~50x faster than slicing
and resampling per window. `rolling_windows_reference` is the plain loop it replaces
and is kept for the equality test in tests/test_harness.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot.backtest import metrics

WINDOW_DAYS = 14
ANN = 365
COLS = ["start", "return", "sharpe", "sortino", "calmar", "max_dd", "score"]
SUMMARY_FIELDS = ("n_windows", "ret_med", "ret_p10", "ret_p90", "p_pos", "p_gt3", "p_gt5",
                  "maxdd_med", "maxdd_p10", "score_med", "score_med_if_pos")  # keys of summarize()


def rolling_windows_reference(equity: pd.Series, days: int = WINDOW_DAYS, step: str = "1D") -> pd.DataFrame:
    """One row per window start (daily steps): competition metrics on that window (slow loop)."""
    eq = equity.dropna()
    starts = pd.date_range(eq.index[0], eq.index[-1] - pd.Timedelta(days=days), freq=step)
    rows = []
    for s in starts:
        seg = eq[s: s + pd.Timedelta(days=days)]
        if len(seg) < 2:
            continue
        m = metrics(seg)
        rows.append({"start": s, **m})
    return pd.DataFrame(rows)


def _window_metrics(seg: np.ndarray, daily: np.ndarray, n_days: float) -> dict:
    """metrics() arithmetic on a window: `seg` hourly equity, `daily` its daily-last values."""
    r = np.diff(daily) / daily[:-1]
    total = seg[-1] / seg[0] - 1
    ann_ret = (1 + total) ** (ANN / max(n_days, 1e-9)) - 1
    sd = float(np.std(r, ddof=1)) if len(r) > 1 else float("nan")
    dd_sd = float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))) if len(r) else float("nan")
    max_dd = float((seg / np.maximum.accumulate(seg) - 1).min())
    mean = float(r.mean()) if len(r) else float("nan")
    sharpe = mean / sd * np.sqrt(ANN) if sd > 0 else 0.0
    sortino = mean / dd_sd * np.sqrt(ANN) if dd_sd > 0 else 0.0
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0
    return {"return": float(total), "sharpe": float(sharpe), "sortino": float(sortino), "calmar": float(calmar),
            "max_dd": max_dd, "score": float(0.4 * sortino + 0.3 * sharpe + 0.3 * calmar)}


def rolling_windows(equity: pd.Series, days: int = WINDOW_DAYS, step: str = "1D") -> pd.DataFrame:
    """One row per window start (daily steps): competition metrics on that window.

    Exactly what `metrics(equity[s : s + days])` gives for each start s, computed fast:
    the daily-last series of the whole curve is built once; a window's daily series is the
    calendar days it covers, with the final (partial) day's value being the window's last
    observation, i.e. the same bins `Series.resample("1D").last()` would produce."""
    eq = equity.dropna()
    if len(eq) < 2:
        return pd.DataFrame(columns=COLS)
    idx, vals = eq.index, eq.to_numpy(dtype=float)
    span = pd.Timedelta(days=days)
    starts = pd.date_range(idx[0], idx[-1] - span, freq=step)
    if len(starts) == 0:
        return pd.DataFrame(columns=COLS)
    day_idx = eq.resample("1D").last().index                      # midnight-anchored, continuous
    last_ts = idx.to_series().resample("1D").last()               # last observation time of each day (NaT if none)
    has_obs = last_ts.notna().to_numpy()
    dvals_full = np.full(len(day_idx), np.nan)
    dvals_full[has_obs] = eq.reindex(last_ts.dropna().to_numpy()).to_numpy(dtype=float)
    # all index arithmetic vectorised over window starts (int64 ns)
    t = idx.asi8
    s_ns, e_ns = starts.asi8, (starts + span).asi8
    day_ns = np.int64(86_400_000_000_000)
    s_day, e_day = (s_ns // day_ns) * day_ns, (e_ns // day_ns) * day_ns
    i0, i1 = t.searchsorted(s_ns, side="left"), t.searchsorted(e_ns, side="right")
    d0, d1 = day_idx.asi8.searchsorted(s_day), day_idx.asi8.searchsorted(e_day)
    dlast_ns = pd.DatetimeIndex(last_ts).asi8                      # NaT -> iNaT (int64 min)
    rows = []
    for k in range(len(starts)):
        a, b = i0[k], i1[k]
        if b - a < 2:
            continue
        seg = vals[a:b]
        daily = dvals_full[d0[k]:d1[k]].copy()
        if daily.size and has_obs[d0[k]] and dlast_ns[d0[k]] < s_ns[k]:
            daily[0] = np.nan                                     # first day's only obs lie before the window
        daily = daily[~np.isnan(daily)]
        if t[b - 1] >= e_day[k]:
            daily = np.append(daily, seg[-1])                     # partial last day: last obs <= window end
        n_days = (t[b - 1] - t[a]) / 86_400_000_000_000
        rows.append({"start": starts[k], **_window_metrics(seg, daily, n_days)})
    return pd.DataFrame(rows, columns=COLS)


def summarize(equity: pd.Series, days: int = WINDOW_DAYS) -> dict:
    """Distribution statistics a 2-week competition cares about."""
    w = rolling_windows(equity, days)
    if w.empty:
        return {}
    r = w["return"]
    return {
        "n_windows": len(w),
        "ret_med": float(r.median()), "ret_p10": float(r.quantile(0.1)), "ret_p90": float(r.quantile(0.9)),
        "p_pos": float((r > 0).mean()), "p_gt3": float((r > 0.03).mean()), "p_gt5": float((r > 0.05).mean()),
        "maxdd_med": float(w["max_dd"].median()), "maxdd_p10": float(w["max_dd"].quantile(0.1)),
        "score_med": float(w["score"].median()),
        "score_med_if_pos": float(w.loc[r > 0, "score"].median()) if (r > 0).any() else float("nan"),
    }
