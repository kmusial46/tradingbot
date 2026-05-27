"""Liquidity engineering (Phase 2 & 5.1).

Pools of resting orders: previous-day high/low, swing highs/lows, and equal
highs/lows (engineered liquidity). A *sweep* is price piercing a pool then closing
back through it. The *draw on liquidity* (DOL) is the pool price is most likely
being delivered toward next.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import pandas as pd

from ..config import StrategyParams
from ..models import LiquidityPool, PoolType, Side, Sweep, SwingPoint, SwingType
from ..timeframes import previous_day_high_low
from .structure import swing_highs, swing_lows


def _cluster_equal(points: List[SwingPoint], tol_pct: float) -> List[List[SwingPoint]]:
    """Group swing points whose prices are within ``tol_pct`` of each other."""
    clusters: List[List[SwingPoint]] = []
    for p in sorted(points, key=lambda s: s.price):
        if clusters and abs(p.price - clusters[-1][-1].price) / max(p.price, 1e-9) <= tol_pct:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return clusters


def liquidity_pools(
    df: pd.DataFrame,
    swings: List[SwingPoint],
    ref_ts: datetime,
    params: StrategyParams,
) -> List[LiquidityPool]:
    """Assemble the liquidity map visible as of ``ref_ts``."""
    pools: List[LiquidityPool] = []

    pdh, pdl = previous_day_high_low(df, ref_ts)
    if pdh is not None:
        pools.append(LiquidityPool(PoolType.BSL, pdh, "PDH"))
    if pdl is not None:
        pools.append(LiquidityPool(PoolType.SSL, pdl, "PDL"))

    highs = swing_highs(swings)
    lows = swing_lows(swings)

    # Equal highs / equal lows (engineered liquidity).
    for cluster in _cluster_equal(highs, params.eqh_eql_tolerance_pct):
        if len(cluster) >= 2:
            price = sum(s.price for s in cluster) / len(cluster)
            pools.append(LiquidityPool(PoolType.BSL, price, "EQH", cluster[-1].ts))
    for cluster in _cluster_equal(lows, params.eqh_eql_tolerance_pct):
        if len(cluster) >= 2:
            price = sum(s.price for s in cluster) / len(cluster)
            pools.append(LiquidityPool(PoolType.SSL, price, "EQL", cluster[-1].ts))

    # Individual swing highs/lows as BSL/SSL.
    for s in highs:
        pools.append(LiquidityPool(PoolType.BSL, s.price, "swing_high", s.ts))
    for s in lows:
        pools.append(LiquidityPool(PoolType.SSL, s.price, "swing_low", s.ts))

    return pools


def detect_sweep(
    df: pd.DataFrame,
    pools: List[LiquidityPool],
    index: int,
    params: StrategyParams,
) -> Optional[Sweep]:
    """Did the bar at ``index`` sweep a pool (wick beyond, close back through)?

    Returns the sweep with the *deepest* pierced pool. Sweeping SSL (lows) creates
    bullish intent; sweeping BSL (highs) creates bearish intent.
    """
    row = df.iloc[index]
    high, low, close = float(row["high"]), float(row["low"]), float(row["close"])
    ts = df.index[index].to_pydatetime()

    best: Optional[Sweep] = None
    best_depth = 0.0
    for pool in pools:
        if pool.type is PoolType.BSL:
            # swept buy-side: pierced above, closed back below
            if high > pool.price and close < pool.price:
                depth = high - pool.price
                if depth > best_depth:
                    best_depth = depth
                    best = Sweep(pool, high, ts, index, Side.SHORT)
        else:  # SSL
            if low < pool.price and close > pool.price:
                depth = pool.price - low
                if depth > best_depth:
                    best_depth = depth
                    best = Sweep(pool, low, ts, index, Side.LONG)
    return best


def draw_on_liquidity(
    pools: List[LiquidityPool],
    price: float,
    bias: Side,
    params: StrategyParams,
) -> Optional[LiquidityPool]:
    """Nearest target pool in the direction of ``bias`` within the max distance.

    Bullish bias draws toward BSL above; bearish toward SSL below.
    """
    want = PoolType.BSL if bias is Side.LONG else PoolType.SSL
    candidates = []
    for p in pools:
        if p.type is not want:
            continue
        if bias is Side.LONG and p.price <= price:
            continue
        if bias is Side.SHORT and p.price >= price:
            continue
        dist_pct = abs(p.price - price) / max(price, 1e-9) * 100.0
        if dist_pct <= params.dol_max_distance_pct:
            candidates.append((abs(p.price - price), p))
    if not candidates:
        return None
    return min(candidates, key=lambda t: t[0])[1]
