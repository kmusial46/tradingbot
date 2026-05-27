"""Higher-timeframe directional bias (Phase 2).

Bias is read from HTF market structure: higher-highs + higher-lows = bullish,
lower-highs + lower-lows = bearish. When structure is ambiguous we fall back to
the sign of the recent close-to-close slope.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from ..config import StrategyParams
from ..models import Bias, Side, SwingPoint
from ..indicators.structure import find_swings, swing_highs, swing_lows


def structural_bias(swings: List[SwingPoint]) -> Bias:
    highs = swing_highs(swings)
    lows = swing_lows(swings)
    if len(highs) >= 2 and len(lows) >= 2:
        hh = highs[-1].price > highs[-2].price
        hl = lows[-1].price > lows[-2].price
        lh = highs[-1].price < highs[-2].price
        ll = lows[-1].price < lows[-2].price
        if hh and hl:
            return Bias.BULLISH
        if lh and ll:
            return Bias.BEARISH
    return Bias.NEUTRAL


def determine_bias(df: pd.DataFrame, params: StrategyParams) -> Bias:
    if df is None or len(df) < 3:
        return Bias.NEUTRAL
    swings = find_swings(df, params.htf_swing_lookback)
    bias = structural_bias(swings)
    if bias is not Bias.NEUTRAL:
        return bias
    # Fallback: slope of close over the recent window.
    n = min(20, len(df) - 1)
    if n <= 0:
        return Bias.NEUTRAL
    change = float(df["close"].iloc[-1] - df["close"].iloc[-1 - n])
    if change > 0:
        return Bias.BULLISH
    if change < 0:
        return Bias.BEARISH
    return Bias.NEUTRAL


def bias_to_side(bias: Bias) -> Optional[Side]:
    if bias is Bias.BULLISH:
        return Side.LONG
    if bias is Bias.BEARISH:
        return Side.SHORT
    return None
