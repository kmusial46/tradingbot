"""SMT divergence (Phase 4): inter-market correlation cracks.

Bullish SMT (at a sell-side pool): the asset prints a Lower Low while its correlated
partner refuses to and prints a Higher Low — institutions are absorbing the sell-off
on the partner. Bearish SMT (at a buy-side pool): the asset prints a Higher High while
the partner prints a Lower High.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import pandas as pd

from ..models import Side, SMTDivergence, SwingPoint, SwingType
from .structure import find_swings, swing_highs, swing_lows


def _last_two(points: List[SwingPoint]) -> Optional[tuple]:
    return (points[-2], points[-1]) if len(points) >= 2 else None


def detect_smt(
    asset_df: pd.DataFrame,
    partner_df: pd.DataFrame,
    side: Side,
    asset: str,
    partner: str,
    lookback: int = 2,
    as_of: Optional[datetime] = None,
) -> Optional[SMTDivergence]:
    """Detect SMT divergence supporting a trade in ``side`` direction.

    Compares the two most recent swing lows (LONG / bullish SMT) or swing highs
    (SHORT / bearish SMT) of the asset versus its partner.
    """
    if as_of is not None:
        asset_df = asset_df[asset_df.index <= as_of]
        partner_df = partner_df[partner_df.index <= as_of]
    if asset_df.empty or partner_df.empty:
        return None

    a_swings = find_swings(asset_df, lookback)
    p_swings = find_swings(partner_df, lookback)

    if side is Side.LONG:
        a = _last_two(swing_lows(a_swings))
        p = _last_two(swing_lows(p_swings))
        if not a or not p:
            return None
        asset_ll = a[1].price < a[0].price          # asset makes a lower low
        partner_hl = p[1].price > p[0].price         # partner holds: higher low
        if asset_ll and partner_hl:
            return SMTDivergence(Side.LONG, asset, partner, a[1].price, p[1].price, a[1].ts)
    else:
        a = _last_two(swing_highs(a_swings))
        p = _last_two(swing_highs(p_swings))
        if not a or not p:
            return None
        asset_hh = a[1].price > a[0].price           # asset makes a higher high
        partner_lh = p[1].price < p[0].price          # partner fails: lower high
        if asset_hh and partner_lh:
            return SMTDivergence(Side.SHORT, asset, partner, a[1].price, p[1].price, a[1].ts)
    return None
