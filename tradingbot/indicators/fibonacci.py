"""Premium / Discount filter (Phase 3).

A Fibonacci is laid over the current dealing range (valid swing low -> swing high).
Above the 50% equilibrium is *premium* (only shorts permitted); below is *discount*
(only longs permitted).
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from ..models import DealingRange, Side, SwingPoint
from .structure import swing_highs, swing_lows


def dealing_range(
    df: pd.DataFrame,
    lookback: int,
    swings: Optional[List[SwingPoint]] = None,
) -> Optional[DealingRange]:
    """Derive the active dealing range from the most recent swing high & low.

    Falls back to the rolling high/low of the window when clean swings are absent.
    """
    if df.empty:
        return None
    window = df.iloc[-lookback:] if lookback and len(df) > lookback else df

    hi_price = lo_price = None
    hi_ts = lo_ts = None
    if swings:
        start_ts = window.index[0]
        highs = [s for s in swing_highs(swings) if s.ts >= start_ts]
        lows = [s for s in swing_lows(swings) if s.ts >= start_ts]
        if highs:
            top = max(highs, key=lambda s: s.price)
            hi_price, hi_ts = top.price, top.ts
        if lows:
            bot = min(lows, key=lambda s: s.price)
            lo_price, lo_ts = bot.price, bot.ts

    if hi_price is None:
        hi_price = float(window["high"].max())
        hi_ts = window["high"].idxmax().to_pydatetime()
    if lo_price is None:
        lo_price = float(window["low"].min())
        lo_ts = window["low"].idxmin().to_pydatetime()

    if hi_price <= lo_price:
        return None
    return DealingRange(low=lo_price, high=hi_price, low_ts=lo_ts, high_ts=hi_ts)


def side_allowed(dr: DealingRange, side: Side, price: float) -> bool:
    """Premium permits only shorts; discount permits only longs."""
    if side is Side.LONG:
        return dr.is_discount(price)
    return dr.is_premium(price)
