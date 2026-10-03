"""Vectorised twin of bot.ml.windows.summarize (same per-window metric definitions as
bot.backtest.metrics, same daily-rolling 14-day windows), ~50x faster so a 900-config
grid is tractable. `check()` asserts equality with the reference on a given curve;
compare_tilts runs it once at start-up before trusting the fast path."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from bot.ml.windows import WINDOW_DAYS, summarize as slow_summarize

ANN = 365


def window_table(equity: pd.Series, days: int = WINDOW_DAYS) -> pd.DataFrame:
    eq = equity.dropna()
    if len(eq) < 2 or (eq.index[-1] - eq.index[0]) < pd.Timedelta(days=days):
        return pd.DataFrame()
    # require a regular hourly grid starting on a day boundary (true for all split frames)
    step = eq.index[1] - eq.index[0]
    per_day = int(pd.Timedelta("1D") / step)
    if not (eq.index.normalize()[0] == eq.index[0] and (eq.index[-1] - eq.index[0]) / step == len(eq) - 1):
        raise ValueError("fast window summary needs a regular grid starting at midnight")
    x = eq.values.astype(float)
    n_win = (len(x) - 1) // per_day - days + 1
    if n_win <= 0:
        return pd.DataFrame()
    hw = sliding_window_view(x, days * per_day + 1)[::per_day][:n_win]   # hourly path of each window
    # bot.backtest.metrics resamples with .resample("1D").last(): the daily points are the LAST
    # observation of each calendar day (23:00) plus the window's final midnight point.
    pts = np.array([per_day - 1 + per_day * k for k in range(days)] + [days * per_day])
    dwin = hw[:, pts]                                        # [n_win, days+1]
    r = dwin[:, 1:] / dwin[:, :-1] - 1.0
    total = hw[:, -1] / hw[:, 0] - 1.0
    ann_ret = (1.0 + total) ** (ANN / days) - 1.0
    sd = r.std(axis=1, ddof=1)
    dd_sd = np.sqrt((np.minimum(r, 0.0) ** 2).mean(axis=1))
    max_dd = (hw / np.maximum.accumulate(hw, axis=1) - 1.0).min(axis=1)
    mean = r.mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sharpe = np.where(sd > 0, mean / sd * np.sqrt(ANN), 0.0)
        sortino = np.where(dd_sd > 0, mean / dd_sd * np.sqrt(ANN), 0.0)
        calmar = np.where(max_dd < 0, ann_ret / np.abs(max_dd), 0.0)
    starts = eq.index[: n_win * per_day: per_day]
    return pd.DataFrame({"start": starts, "return": total, "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
                         "max_dd": max_dd, "score": 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar})


def summarize(equity: pd.Series, days: int = WINDOW_DAYS) -> dict:
    w = window_table(equity, days)
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


def check(equity: pd.Series, tol: float = 1e-7) -> None:
    a, b = slow_summarize(equity), summarize(equity)
    assert a.keys() == b.keys(), (a.keys(), b.keys())
    for k in a:
        if not (np.isnan(a[k]) and np.isnan(b[k])) and abs(a[k] - b[k]) > tol * max(1.0, abs(a[k])):
            raise AssertionError(f"fast summarize mismatch on {k}: {a[k]} vs {b[k]}")
