"""Core domain types shared across the bot.

Time-series data is carried in pandas DataFrames (see ``timeframes.py``) for
vectorized detection. The dataclasses here represent *detected structures* and
*outputs* (swings, FVGs, sweeps, signals, trades) plus a lightweight ``Bar`` for
the incremental live path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"

    @property
    def opposite(self) -> "Side":
        return Side.SHORT if self is Side.LONG else Side.LONG


class Bias(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SwingType(str, Enum):
    HIGH = "high"
    LOW = "low"


class PoolType(str, Enum):
    """Buy-side liquidity rests above price (old highs); sell-side below (old lows)."""

    BSL = "bsl"  # buy-side liquidity -> resting buy stops above highs
    SSL = "ssl"  # sell-side liquidity -> resting sell stops below lows


class TradeOutcome(str, Enum):
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    OPEN = "open"


@dataclass(frozen=True)
class Bar:
    """A single OHLCV candle. ``ts`` is the bar's open time, tz-aware (NY)."""

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def body_ratio(self) -> float:
        """Body as a fraction of total range (1.0 = marubozu, 0.0 = doji)."""
        return self.body / self.range if self.range > 0 else 0.0

    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2.0


@dataclass(frozen=True)
class SwingPoint:
    ts: datetime
    price: float
    type: SwingType
    index: int  # positional index within the source frame


@dataclass(frozen=True)
class FVG:
    """A 3-candle fair value gap (imbalance).

    Bullish FVG: gap between candle-1 high and candle-3 low (price ran up).
    Bearish FVG: gap between candle-1 low and candle-3 high (price ran down).
    ``top``/``bottom`` bound the empty zone; ``entry`` is the boundary a limit
    order is parked at (proximal edge a retracement reaches first).
    """

    side: Side  # LONG = bullish FVG (support), SHORT = bearish FVG (resistance)
    top: float
    bottom: float
    ts: datetime
    index: int  # index of the middle (displacement) candle

    @property
    def size(self) -> float:
        return self.top - self.bottom

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def entry(self) -> float:
        """Proximal boundary: a retrace into a bullish FVG enters at its top,
        into a bearish FVG at its bottom."""
        return self.top if self.side is Side.LONG else self.bottom

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top


@dataclass(frozen=True)
class LiquidityPool:
    type: PoolType
    price: float
    label: str  # e.g. "PDH", "PDL", "EQH", "EQL", "swing_high", "swing_low"
    ts: Optional[datetime] = None

    @property
    def is_buy_side(self) -> bool:
        return self.type is PoolType.BSL


@dataclass(frozen=True)
class Sweep:
    """A liquidity sweep: price pierced a pool then rejected back through it."""

    pool: LiquidityPool
    extreme: float  # the high/low printed during the sweep (stop placement ref)
    ts: datetime
    index: int
    # The directional intent created by the sweep. Sweeping SSL (lows) is bullish
    # intent; sweeping BSL (highs) is bearish intent.
    intent: Side


@dataclass(frozen=True)
class DealingRange:
    low: float
    high: float
    low_ts: Optional[datetime] = None
    high_ts: Optional[datetime] = None

    @property
    def equilibrium(self) -> float:
        return (self.high + self.low) / 2.0

    def position(self, price: float) -> float:
        """Fractional position of ``price`` in the range. 0 = low, 1 = high."""
        span = self.high - self.low
        if span <= 0:
            return 0.5
        return (price - self.low) / span

    def is_premium(self, price: float) -> bool:
        return self.position(price) > 0.5

    def is_discount(self, price: float) -> bool:
        return self.position(price) < 0.5


@dataclass(frozen=True)
class SMTDivergence:
    """Inter-market divergence between an asset and its correlated partner."""

    side: Side  # LONG = bullish SMT (at SSL), SHORT = bearish SMT (at BSL)
    asset: str
    partner: str
    asset_extreme: float
    partner_extreme: float
    ts: datetime


@dataclass
class Signal:
    """A fully-formed trade setup ready to be acted on (alerted, not auto-traded)."""

    ts: datetime
    symbol: str
    side: Side
    entry: float
    stop: float
    targets: List[float]
    killzone: str
    bias: Bias
    reasons: List[str] = field(default_factory=list)
    fvg: Optional[FVG] = None
    sweep: Optional[Sweep] = None
    smt: Optional[SMTDivergence] = None
    rr: Optional[float] = None  # planned reward:risk to the first target

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry - self.stop)

    def reward_risk(self, target: Optional[float] = None) -> float:
        tgt = target if target is not None else (self.targets[0] if self.targets else self.entry)
        risk = self.risk_per_unit
        if risk <= 0:
            return 0.0
        return abs(tgt - self.entry) / risk


@dataclass
class Trade:
    """A simulated/realized trade derived from a Signal during backtesting."""

    signal: Signal
    symbol: str
    side: Side
    entry_ts: datetime
    entry: float
    stop: float
    target: float
    size: float = 0.0
    risk_amount: float = 0.0
    exit_ts: Optional[datetime] = None
    exit: Optional[float] = None
    outcome: TradeOutcome = TradeOutcome.OPEN
    r_multiple: float = 0.0
    moved_to_be: bool = False
    pnl: float = 0.0
    notes: str = ""
