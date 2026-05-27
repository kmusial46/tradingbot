"""Configuration: risk, detection thresholds, instruments, timeframes, provider.

All knobs have sane defaults so the bot runs with no config file. ``Config.load``
overlays a YAML file on top of the defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class RiskParams:
    account_size: float = 100_000.0
    risk_pct: float = 1.0          # percent of account risked per trade (1-2%)
    rr_target: float = 2.0         # mechanical reward:risk for the primary target
    second_target_rr: float = 3.0  # optional runner target
    move_be_at_r: float = 1.0      # move stop to breakeven once price reaches this R


@dataclass
class StrategyParams:
    # Swing / structure detection
    swing_lookback: int = 2            # fractal half-width on the execution TF
    htf_swing_lookback: int = 2        # fractal half-width on HTF for bias/DOL

    # Dealing range (premium/discount)
    dealing_range_lookback: int = 240        # exec-TF bars (unused when HTF range on)
    htf_dealing_range_lookback: int = 20     # HTF bars for the premium/discount range
    require_premium_discount: bool = True     # gate entries by premium/discount zone

    # Liquidity
    eqh_eql_tolerance_pct: float = 0.03   # |a-b|/price under this => "equal"
    sweep_lookback: int = 60              # bars to scan for the swept pool
    dol_max_distance_pct: float = 1.5     # ignore pools farther than this from price

    # Displacement (institutional candle)
    displacement_atr_mult: float = 1.5    # body must exceed this * ATR
    displacement_body_ratio: float = 0.55 # body / range minimum
    atr_period: int = 14

    # Fair value gap
    fvg_min_size_atr: float = 0.10        # min gap size as a fraction of ATR

    # SMT
    smt_swing_lookback: int = 2

    # Market structure shift confirmation window (bars after sweep)
    mss_max_bars: int = 20

    # Require an active 20-min macro window in addition to the killzone?
    require_macro: bool = False


@dataclass
class Instrument:
    symbol: str                    # canonical token, e.g. "NQ"
    tv_symbol: str = ""            # TradingView symbol for the MCP, e.g. "CME_MINI:NQ1!"
    smt_partner: Optional[str] = None  # canonical token of correlated partner, e.g. "ES"
    enabled: bool = True


@dataclass
class TimeframeParams:
    execution: str = "5min"            # entry timeframe (1min or 5min)
    htf: List[str] = field(default_factory=lambda: ["1h", "4h"])
    bias_tf: str = "1h"                # timeframe used to derive directional bias
    dol_tf: str = "1h"                 # timeframe used to mark draw-on-liquidity pools


@dataclass
class BacktestParams:
    window: int = 720             # exec bars passed to the engine each step
    warmup_bars: int = 60         # skip the first N bars (insufficient context)
    pending_expiry_bars: int = 30  # cancel an unfilled limit after this many bars
    one_trade_per_symbol: bool = True  # at most one live/pending trade per instrument


@dataclass
class ProviderConfig:
    type: str = "synthetic"            # synthetic | csv | tv_mcp
    # csv provider
    csv_dir: str = "data"
    # tv_mcp provider (local TradingView Desktop + tradesdontlie/tradingview-mcp)
    mcp_command: str = "node"
    mcp_args: List[str] = field(default_factory=lambda: ["/path/to/tradingview-mcp/src/server.js"])
    mcp_bars_per_call: int = 100       # data_get_ohlcv full-mode page size
    # synthetic provider
    synthetic_seed: int = 7
    synthetic_days: int = 5


@dataclass
class Config:
    risk: RiskParams = field(default_factory=RiskParams)
    strategy: StrategyParams = field(default_factory=StrategyParams)
    timeframes: TimeframeParams = field(default_factory=TimeframeParams)
    backtest: BacktestParams = field(default_factory=BacktestParams)
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    instruments: List[Instrument] = field(default_factory=lambda: [
        Instrument("NQ", "CME_MINI:NQ1!", smt_partner="ES"),
        Instrument("ES", "CME_MINI:ES1!", smt_partner="NQ"),
        Instrument("GOLD", "COMEX:GC1!", smt_partner=None),
    ])
    # Which killzones to hunt in (by name). Empty => all.
    enabled_killzones: List[str] = field(default_factory=list)

    # ---- loading -------------------------------------------------------

    @classmethod
    def default(cls) -> "Config":
        return cls()

    @classmethod
    def load(cls, path: Optional[str]) -> "Config":
        if not path:
            return cls.default()
        if yaml is None:  # pragma: no cover
            raise RuntimeError("PyYAML is required to load a config file")
        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        cfg = cls.default()
        _overlay(cfg.risk, data.get("risk", {}))
        _overlay(cfg.strategy, data.get("strategy", {}))
        _overlay(cfg.timeframes, data.get("timeframes", {}))
        _overlay(cfg.backtest, data.get("backtest", {}))
        _overlay(cfg.provider, data.get("provider", {}))
        if "instruments" in data and data["instruments"]:
            cfg.instruments = [Instrument(**i) for i in data["instruments"]]
        if "enabled_killzones" in data:
            cfg.enabled_killzones = list(data["enabled_killzones"])
        return cfg

    def instrument(self, symbol: str) -> Optional[Instrument]:
        for ins in self.instruments:
            if ins.symbol == symbol:
                return ins
        return None


def _overlay(target: Any, updates: Dict[str, Any]) -> None:
    """Shallow-overlay a dict of values onto a dataclass instance, in place."""
    if not updates:
        return
    valid = {f.name for f in fields(target)} if is_dataclass(target) else set()
    for key, value in updates.items():
        if key in valid:
            setattr(target, key, value)
