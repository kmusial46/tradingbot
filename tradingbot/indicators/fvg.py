"""Fair Value Gaps (Phase 5.4): 3-candle imbalances left by a displacement wave.

Bullish FVG: ``low[i+1] > high[i-1]`` — the empty band (high[i-1] .. low[i+1]) is
unfilled buy-side inefficiency; a retrace into its top is the long entry.
Bearish FVG: ``high[i+1] < low[i-1]`` — the band (high[i+1] .. low[i-1]) is sell-side
inefficiency; a retrace into its bottom is the short entry.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from ..models import FVG, Side


def find_fvgs(df: pd.DataFrame, side: Optional[Side] = None, min_size: float = 0.0) -> List[FVG]:
    """All FVGs in ``df``. The middle (displacement) candle is index ``i``."""
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    idx = df.index
    out: List[FVG] = []
    for i in range(1, len(df) - 1):
        # bullish gap
        if lows[i + 1] > highs[i - 1]:
            gap = FVG(Side.LONG, float(lows[i + 1]), float(highs[i - 1]), idx[i].to_pydatetime(), i)
            if gap.size >= min_size and (side is None or side is Side.LONG):
                out.append(gap)
        # bearish gap
        elif highs[i + 1] < lows[i - 1]:
            gap = FVG(Side.SHORT, float(lows[i - 1]), float(highs[i + 1]), idx[i].to_pydatetime(), i)
            if gap.size >= min_size and (side is None or side is Side.SHORT):
                out.append(gap)
    return out


def fvg_in_leg(
    df: pd.DataFrame,
    side: Side,
    lo: int,
    hi: int,
    min_size: float = 0.0,
) -> Optional[FVG]:
    """The FVG of ``side`` within the displacement leg [lo, hi], nearest to the leg end.

    This is the entry frame: the imbalance the displacement wave left behind.
    Returns the most recent qualifying gap (closest to current price).
    """
    candidates = [
        g for g in find_fvgs(df, side=side, min_size=min_size)
        if lo <= g.index <= hi
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda g: g.index)
