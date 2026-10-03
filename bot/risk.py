"""Portfolio-level risk overlay shared by backtest and live trading.

Drawdown throttle: as equity falls from its recent peak we cut gross exposure,
which directly caps max drawdown (the Calmar denominator) and the downside
deviation that Sortino punishes.

The peak is measured over a trailing window equal to the competition length
(14 days). An all-time peak would leave the book permanently throttled after
one bad month in a multi-year backtest, which is not how a 2-week competition
behaves; a rolling peak matches the horizon the score is computed on.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class RiskParams:
    dd_soft: float = 0.05   # start de-risking at -5% from the rolling peak
    dd_hard: float = 0.10   # minimum exposure from -10% onward
    min_scale: float = 0.30
    window_h: int = 24 * 14  # rolling peak window (hours)


def exposure_scale(drawdown: float, rp: RiskParams = RiskParams()) -> float:
    """drawdown <= 0 (e.g. -0.05). Linear ramp from 1.0 at dd_soft to min_scale at dd_hard."""
    dd = -drawdown
    if dd <= rp.dd_soft:
        return 1.0
    if dd >= rp.dd_hard:
        return rp.min_scale
    frac = (dd - rp.dd_soft) / (rp.dd_hard - rp.dd_soft)
    return 1.0 - frac * (1.0 - rp.min_scale)


class RollingPeak:
    """Tracks the max equity over the last `window_h` hourly observations."""

    def __init__(self, window_h: int):
        self.buf: deque[float] = deque(maxlen=window_h)

    def update(self, equity: float) -> float:
        self.buf.append(equity)
        return max(self.buf)
