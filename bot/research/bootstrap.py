"""Paired block bootstrap for rolling 14-day window statistics.

Two strategies' daily returns are resampled jointly (same circular blocks, so
market regimes stay aligned between the two), equity is rebuilt, and the rolling
14-day window statistics (median window return, share of positive windows) are
recomputed on each pseudo-sample. The distribution of the difference
(candidate - reference) gives a confidence interval that accounts for the heavy
overlap of daily-rolling windows, which the raw window count ignores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW_D = 14


def daily_equity(equity: pd.Series) -> pd.Series:
    return equity.resample("1D").last().dropna()


def window_stats(daily_rets: np.ndarray, window: int = WINDOW_D) -> tuple[float, float]:
    """(median 14d return, P(14d return > 0)) over all daily-rolling windows of a daily return path."""
    lg = np.log1p(daily_rets)
    c = np.concatenate([[0.0], np.cumsum(lg)])
    wr = np.exp(c[window:] - c[:-window]) - 1.0
    return float(np.median(wr)), float((wr > 0).mean())


def _blocks(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block bootstrap indices of length n."""
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
    return idx[:n]


def paired_bootstrap(ref: pd.Series, cand: pd.Series, n_boot: int = 1000, block: int = WINDOW_D,
                     seed: int = 0) -> dict:
    """ref/cand: hourly or daily equity Series on the same period. Returns point estimates
    and percentile CIs of (cand - ref) for ret_med and p_pos, plus P(cand > ref)."""
    a, b = daily_equity(ref), daily_equity(cand)
    ix = a.index.intersection(b.index)
    ra = a.reindex(ix).pct_change().dropna().values
    rb = b.reindex(ix).pct_change().dropna().values
    n = len(ra)
    rng = np.random.default_rng(seed)
    d_ret, d_pos = np.empty(n_boot), np.empty(n_boot)
    for k in range(n_boot):
        idx = _blocks(n, block, rng)
        m_a, p_a = window_stats(ra[idx])
        m_b, p_b = window_stats(rb[idx])
        d_ret[k], d_pos[k] = m_b - m_a, p_b - p_a
    m_a, p_a = window_stats(ra)
    m_b, p_b = window_stats(rb)
    return {
        "ref_ret_med": m_a, "cand_ret_med": m_b, "d_ret_med": m_b - m_a,
        "d_ret_lo": float(np.percentile(d_ret, 2.5)), "d_ret_hi": float(np.percentile(d_ret, 97.5)),
        "p_ret_gt": float((d_ret > 0).mean()),
        "ref_p_pos": p_a, "cand_p_pos": p_b, "d_p_pos": p_b - p_a,
        "d_pos_lo": float(np.percentile(d_pos, 2.5)), "d_pos_hi": float(np.percentile(d_pos, 97.5)),
        "p_pos_gt": float((d_pos > 0).mean()),
        "n_days": n, "n_boot": n_boot, "block_d": block,
    }
