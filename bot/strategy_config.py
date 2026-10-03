"""Declarative strategy configuration for the live bot.

One JSON file (config/strategy.json by default) fully describes what the bot
trades, so a strategy change during the competition is a config edit plus a
commit. The fields mirror the backtest objects they feed:

  core / core_frac / ema_span_h / core_vol / core_corr / short_core
      -> bot.research.tilts.CoreTiltWeights (EMA-gated, vol-targeted BTC/ETH core
         with the optional BTC short overlay); bot.ml.live_signal is its live port
  tilt / tilt_params
      -> bot.ml.portfolio.PortParams used by weights_from_scores for the tilt
         ("cnn" | "xs_mom" | "trend" | "none")
  models   -> CNN checkpoints in models/<tag>.pt, rank-averaged when > 1
  universe -> "crypto" (scored; Binance history exists) or "all"
  risk     -> bot.risk.RiskParams (drawdown throttle)
  execution-> limit-first execution settings (bot.execution)
  kind     -> "core_tilt" (the fields above) or "risk_parity": the Sharpe-first
              risk-balanced trend portfolio in bot.strategies.risk_parity, configured by
              the `risk_parity` section (RPParams); it rebalances once a day at 00:xx UTC.

Unknown keys are rejected so a typo cannot silently fall back to a default.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from bot.ml.portfolio import PortParams
from bot.strategies.risk_parity import RPParams
from bot.risk import RiskParams

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "config" / "strategy.json"
TILTS = ("none", "cnn", "xs_mom", "trend")
UNIVERSES = ("crypto", "all")
KINDS = ("core_tilt", "risk_parity")


@dataclass
class ExecutionParams:
    limit_timeout_s: int = 180       # cancel the resting limit and go market after this
    order_gap_s: int = 61            # at most one new order per minute (competition pacing)
    poll_s: int = 5                  # query_order polling interval while a limit rests
    min_trade_usd: float = 25.0      # ignore smaller rebalance legs
    fee_buffer: float = 0.995        # buys use at most this fraction of free USD


@dataclass
class StrategyConfig:
    name: str = "core_tilt"
    kind: str = "core_tilt"
    risk_parity: RPParams = field(default_factory=RPParams)
    core: list[str] = field(default_factory=lambda: ["BTC", "ETH"])
    core_frac: float = 0.7
    ema_span_h: int = 1200           # 50 days of hourly bars
    core_vol: float = 0.6            # annualised vol target for the core sleeve
    core_corr: float = 0.8           # assumed pairwise correlation among core coins (vol targeting)
    short_core: bool = False         # BTC short overlay: when BTC < EMA the whole core budget shorts BTC (1x)
    tilt: str = "cnn"
    tilt_params: PortParams = field(default_factory=PortParams)
    models: list[str] = field(default_factory=lambda: ["cnn64", "cnn32"])
    universe: str = "crypto"
    risk: RiskParams = field(default_factory=RiskParams)
    execution: ExecutionParams = field(default_factory=ExecutionParams)

    def validate(self) -> "StrategyConfig":
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if self.kind == "risk_parity":
            self.risk_parity.validate()
            return self
        if self.tilt not in TILTS:
            raise ValueError(f"tilt must be one of {TILTS}, got {self.tilt!r}")
        if self.universe not in UNIVERSES:
            raise ValueError(f"universe must be one of {UNIVERSES}, got {self.universe!r}")
        if not 0.0 <= self.core_frac <= 1.0:
            raise ValueError("core_frac must be in [0, 1]")
        if self.core_frac > 0 and not self.core:
            raise ValueError("core_frac > 0 needs at least one core coin")
        if self.tilt == "cnn" and self.core_frac < 1.0 and not self.models:
            raise ValueError("tilt 'cnn' needs at least one model tag")
        if self.ema_span_h < 2 or self.core_vol <= 0:
            raise ValueError("ema_span_h must be >= 2 and core_vol > 0")
        if not -1.0 <= self.core_corr <= 1.0:
            raise ValueError("core_corr must be in [-1, 1]")
        if self.short_core and "BTC" not in self.core:
            raise ValueError("short_core needs BTC in core")
        if self.tilt_params.rebal_h < 1 or self.tilt_params.top_k < 1:
            raise ValueError("rebal_h and top_k must be >= 1")
        return self

    @property
    def rebal_h(self) -> int:
        return int(self.risk_parity.rebal_h) if self.kind == "risk_parity" else int(self.tilt_params.rebal_h)

    @property
    def band(self) -> float:
        return self.risk_parity.band if self.kind == "risk_parity" else self.tilt_params.band

    def to_dict(self) -> dict:
        return asdict(self)


def _build(cls, data: dict, where: str):
    allowed = {f.name for f in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown keys in {where}: {sorted(unknown)} (allowed: {sorted(allowed)})")
    return cls(**data)


def from_dict(d: dict) -> StrategyConfig:
    d = dict(d)
    nested = {"tilt_params": PortParams, "risk": RiskParams, "execution": ExecutionParams,
              "risk_parity": RPParams}
    for key, cls in nested.items():
        if key in d:
            d[key] = _build(cls, d[key] or {}, key)
    return _build(StrategyConfig, d, "strategy").validate()


def load_strategy(path: str | Path | None = None) -> StrategyConfig:
    p = Path(path) if path else DEFAULT_PATH
    return from_dict(json.loads(p.read_text()))


def save_strategy(cfg: StrategyConfig, path: str | Path | None = None) -> Path:
    p = Path(path) if path else DEFAULT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    return p


if __name__ == "__main__":  # print the active config (validates it)
    import sys

    print(json.dumps(load_strategy(sys.argv[1] if len(sys.argv) > 1 else None).to_dict(), indent=2))
