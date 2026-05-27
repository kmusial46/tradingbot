"""Synthetic OHLC generator + provider.

Produces deterministic, correlated 1-minute data for NQ/ES/GOLD with *scripted*
ICT setups injected inside killzones: liquidity sweeps, displacement legs that
leave fair-value gaps and shift structure, and inter-market (SMT) divergences
between NQ and ES. This lets the full strategy + backtester run and be validated
anywhere, with no live feed. It is NOT a market simulator — it is a fixture.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from ..timeutils import NY
from ..timeframes import normalize_frame, resample
from .base import DataProvider

# Per-symbol amplitude unit (A) in price points and a starting price.
_SYMBOL_SPEC = {
    "NQ": (20.0, 18000.0),
    "ES": (5.0, 5300.0),
    "GOLD": (4.0, 2350.0),
}

# A scripted setup pattern, expressed as (open, high, low, close) offsets in units
# of A relative to the pre-episode price `b`. "asset" makes the extreme that gets
# swept (LL for bullish, HH for bearish); "partner" refuses to (HL / LH) -> SMT.
def _bullish_pattern(role: str) -> List[Tuple[float, float, float, float]]:
    sweep_low = -3.0 if role == "asset" else -2.0  # partner holds a higher low
    return [
        (0.00, 0.50, -0.50, 0.20),
        (0.20, 0.40, -0.30, -0.10),
        (-0.10, 0.20, -0.60, -0.40),
        (-0.40, -0.20, -2.50, -1.20),   # 3: swing low
        (-1.20, -0.60, -1.40, -0.80),
        (-0.80, 0.30, -0.90, 0.10),
        (0.10, 1.00, -0.10, 0.70),      # 6: swing high (MSS reference)
        (0.70, 0.80, -0.30, -0.10),
        (-0.10, 0.10, -1.20, -0.90),
        (-0.90, -0.60, sweep_low, -1.50),  # 9: sweep of the low
        (-1.50, 1.50, -1.70, 1.20),     # 10: displacement up
        (1.20, 3.20, 1.00, 3.00),       # 11: displacement up -> MSS (close > +1.0A)
        (3.00, 3.40, 2.40, 2.60),
        (2.60, 2.80, 0.90, 1.10),       # 13: retrace into FVG (top ~ +1.0A)
        (1.10, 2.50, 1.00, 2.30),
        (2.30, 3.80, 2.20, 3.60),
        (3.60, 5.20, 3.40, 5.00),
        (5.00, 6.80, 4.80, 6.60),
        (6.60, 8.40, 6.40, 8.20),
        (8.20, 9.80, 8.00, 9.60),       # run reaches +2R region
        (9.60, 10.4, 9.20, 9.80),
        (9.80, 10.0, 9.00, 9.40),
        (9.40, 9.80, 8.80, 9.20),
        (9.20, 9.60, 8.60, 9.00),
    ]


def _bearish_pattern(role: str) -> List[Tuple[float, float, float, float]]:
    return [(-o, -lo, -hi, -c) for (o, hi, lo, c) in _bullish_pattern(role)]


def _pattern(direction: str, role: str) -> List[Tuple[float, float, float, float]]:
    return _bullish_pattern(role) if direction == "bullish" else _bearish_pattern(role)


def _gen_day(
    rng: np.random.Generator,
    day: date,
    start_price: float,
    amp: float,
    drift: float,
    episodes: Dict[int, Tuple[str, str]],
) -> pd.DataFrame:
    """Generate one NY day of 1-minute bars with scripted episodes."""
    n = 1440
    sigma = 0.15 * amp
    o = np.empty(n); h = np.empty(n); l = np.empty(n); c = np.empty(n)
    i = 0
    while i < n:
        if i in episodes and i + 24 <= n:
            direction, role = episodes[i]
            b = c[i - 1] if i > 0 else start_price
            for k, (oo, hh, ll, cc) in enumerate(_pattern(direction, role)):
                o[i + k] = b + oo * amp
                h[i + k] = b + hh * amp
                l[i + k] = b + ll * amp
                c[i + k] = b + cc * amp
            i += 24
        else:
            op = c[i - 1] if i > 0 else start_price
            cl = op + rng.normal(drift, sigma)
            hi = max(op, cl) + abs(rng.normal(0, sigma * 0.6))
            lo = min(op, cl) - abs(rng.normal(0, sigma * 0.6))
            o[i], h[i], l[i], c[i] = op, hi, lo, cl
            i += 1

    vol = rng.uniform(800, 1500, n)
    idx = pd.date_range(datetime(day.year, day.month, day.day, 0, 0, tzinfo=NY), periods=n, freq="1min")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": vol}, index=idx)


def _weekdays_ending(end: date, days: int) -> List[date]:
    out: List[date] = []
    d = end
    while len(out) < days:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return list(reversed(out))


def generate_dataset(
    symbols: List[str],
    end_date: Optional[date] = None,
    days: int = 5,
    seed: int = 7,
) -> Dict[str, pd.DataFrame]:
    """Generate correlated 1-minute frames for ``symbols`` over ``days`` weekdays."""
    if end_date is None:
        end_date = datetime.now(NY).date() - timedelta(days=1)
    trading_days = _weekdays_ending(end_date, days)
    rng = np.random.default_rng(seed)

    # Up-trend first half of the week, down-trend second half, so HTF bias aligns
    # with the injected episode direction inside each segment.
    half = len(trading_days) / 2.0

    frames: Dict[str, pd.DataFrame] = {sym: [] for sym in symbols}
    last_price = {sym: _SYMBOL_SPEC.get(sym, (5.0, 1000.0))[1] for sym in symbols}

    for di, day in enumerate(trading_days):
        bullish = di < half
        direction = "bullish" if bullish else "bearish"
        drift_sign = 1.0 if bullish else -1.0
        # Killzone minute anchors (NY): 07:30 NY Open, 10:15 Silver Bullet, 14:30 PM.
        if bullish:
            anchors = [7 * 60 + 30, 10 * 60 + 15]
        else:
            anchors = [7 * 60 + 30, 14 * 60 + 30]

        for sym in symbols:
            amp, _ = _SYMBOL_SPEC.get(sym, (5.0, 1000.0))
            role = "partner" if sym == "ES" else "asset"
            episodes = {a: (direction, role) for a in anchors}
            day_df = _gen_day(rng, day, last_price[sym], amp, drift_sign * 0.03 * amp, episodes)
            frames[sym].append(day_df)
            last_price[sym] = float(day_df["close"].iloc[-1])

    return {sym: normalize_frame(pd.concat(parts)) for sym, parts in frames.items()}


class SyntheticProvider(DataProvider):
    name = "synthetic"

    def __init__(self, config: Config, end_date: Optional[date] = None):
        self.config = config
        days = getattr(config.provider, "synthetic_days", 5)
        symbols = [i.symbol for i in config.instruments]
        self._frames = generate_dataset(
            symbols, end_date=end_date, days=days, seed=config.provider.synthetic_seed
        )

    def get_history(self, symbol, timeframe, start=None, end=None, limit=None):
        base = self._frames.get(symbol)
        if base is None:
            raise KeyError(f"no synthetic data for {symbol!r}")
        df = base if timeframe in ("1min", "1m") else resample(base, timeframe)
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index <= end]
        if limit is not None:
            df = df.iloc[-limit:]
        return df.copy()
