"""Displacement: wide-bodied, low-wick candles signalling institutional intent.

Phase 5.3 — the move that breaks structure must be driven by displacement, not a
slow, choppy grind. We qualify a candle by body size relative to ATR and by its
body-to-range ratio.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from ..config import StrategyParams
from ..models import Side


def is_displacement(row: pd.Series, atr_value: float, params: StrategyParams, side: Optional[Side] = None) -> bool:
    body = abs(row["close"] - row["open"])
    rng = row["high"] - row["low"]
    if rng <= 0 or atr_value <= 0:
        return False
    body_ratio = body / rng
    if body < params.displacement_atr_mult * atr_value:
        return False
    if body_ratio < params.displacement_body_ratio:
        return False
    if side is Side.LONG and row["close"] <= row["open"]:
        return False
    if side is Side.SHORT and row["close"] >= row["open"]:
        return False
    return True


def find_displacement(
    df: pd.DataFrame,
    atr_series: pd.Series,
    params: StrategyParams,
    lo: int,
    hi: int,
    side: Optional[Side] = None,
) -> Optional[int]:
    """Index of the strongest qualifying displacement candle in [lo, hi] (inclusive)."""
    n = len(df)
    lo = max(0, lo)
    hi = min(n - 1, hi)
    best_idx: Optional[int] = None
    best_body = 0.0
    atr_vals = atr_series.to_numpy()
    for i in range(lo, hi + 1):
        row = df.iloc[i]
        if is_displacement(row, float(atr_vals[i]), params, side):
            body = abs(row["close"] - row["open"])
            if body > best_body:
                best_body = body
                best_idx = i
    return best_idx
