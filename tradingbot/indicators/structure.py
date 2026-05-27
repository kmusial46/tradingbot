"""Market structure: swing pivots and the Market Structure Shift (MSS).

A swing high/low is a fractal pivot — an extreme that is strictly more extreme
than the ``lookback`` bars on each side. The MSS (Phase 5.2) confirms a directional
shift: after a sweep, price must *close* cleanly past the most recent opposing
structural swing.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from ..models import Side, SwingPoint, SwingType


def find_swings(df: pd.DataFrame, lookback: int = 2) -> List[SwingPoint]:
    """Return fractal swing points (strict pivots) ordered by position."""
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    idx = df.index
    n = len(df)
    out: List[SwingPoint] = []
    if n < 2 * lookback + 1:
        return out
    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback:i + lookback + 1]
        window_l = lows[i - lookback:i + lookback + 1]
        if highs[i] == window_h.max() and (window_h == highs[i]).sum() == 1:
            out.append(SwingPoint(idx[i].to_pydatetime(), float(highs[i]), SwingType.HIGH, i))
        elif lows[i] == window_l.min() and (window_l == lows[i]).sum() == 1:
            out.append(SwingPoint(idx[i].to_pydatetime(), float(lows[i]), SwingType.LOW, i))
    return out


def swing_highs(swings: List[SwingPoint]) -> List[SwingPoint]:
    return [s for s in swings if s.type is SwingType.HIGH]


def swing_lows(swings: List[SwingPoint]) -> List[SwingPoint]:
    return [s for s in swings if s.type is SwingType.LOW]


def last_swing_before(swings: List[SwingPoint], type_: SwingType, index: int) -> Optional[SwingPoint]:
    """Most recent swing of ``type_`` at or before positional ``index``."""
    cand = [s for s in swings if s.type is type_ and s.index <= index]
    return cand[-1] if cand else None


@dataclass(frozen=True)
class MSS:
    """A confirmed Market Structure Shift."""

    side: Side                 # direction of the shift (LONG = bullish)
    broken_swing: SwingPoint   # the opposing structural swing that was broken
    break_index: int           # positional index of the bar that closed past it
    break_ts: datetime
    break_close: float


def detect_mss(
    df: pd.DataFrame,
    swings: List[SwingPoint],
    side: Side,
    after_index: int,
    max_bars: int = 20,
) -> Optional[MSS]:
    """Detect an MSS in ``side`` direction within ``max_bars`` after ``after_index``.

    LONG: price must close *above* the most recent swing high formed at/before the
    sweep. SHORT: price must close *below* the most recent swing low.
    """
    n = len(df)
    if side is Side.LONG:
        ref = last_swing_before(swings, SwingType.HIGH, after_index)
    else:
        ref = last_swing_before(swings, SwingType.LOW, after_index)
    if ref is None:
        return None

    closes = df["close"].to_numpy()
    idx = df.index
    hi = min(n, after_index + max_bars + 1)
    for i in range(after_index + 1, hi):
        if side is Side.LONG and closes[i] > ref.price:
            return MSS(Side.LONG, ref, i, idx[i].to_pydatetime(), float(closes[i]))
        if side is Side.SHORT and closes[i] < ref.price:
            return MSS(Side.SHORT, ref, i, idx[i].to_pydatetime(), float(closes[i]))
    return None
