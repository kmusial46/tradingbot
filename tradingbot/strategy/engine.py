"""The strategy engine: Phase 1-6 orchestration into a single Signal.

Given a multi-timeframe ``MarketContext`` for one instrument as of a point in time,
``StrategyEngine.evaluate`` returns a ``Signal`` iff every gate aligns:

  1. Time      -- inside an enabled killzone (and optionally an algo macro).
  2. Bias/DOL  -- HTF structure gives a directional bias and side.
  5.1 Sweep    -- a recent sweep of a liquidity pool with matching intent.
  5.2 MSS      -- price closes past the opposing swing on the current bar.
  5.3 Displace -- the breaking leg contains an institutional displacement candle.
  5.4 FVG      -- the leg left a fair-value gap (the entry frame).
  3. Filter    -- the FVG entry sits in the correct premium/discount zone.
  4. SMT       -- inter-market divergence confirms (when a partner is configured).
  6. Risk      -- stop at the sweep extreme; mechanical R-multiple targets.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import pandas as pd

from ..config import Config
from ..models import LiquidityPool, Side, Signal
from ..timeutils import active_killzones, in_macro, ny_midnight
from ..timeframes import atr
from ..indicators.structure import find_swings, detect_mss
from ..indicators.displacement import find_displacement
from ..indicators.fvg import find_fvgs
from ..indicators.liquidity import liquidity_pools, detect_sweep
from ..indicators.fibonacci import dealing_range, side_allowed
from ..indicators.smt import detect_smt
from .bias import determine_bias, bias_to_side


@dataclass
class MarketContext:
    """A snapshot of the data available for one instrument as of ``now``.

    All frames must already be sliced so they contain no information after ``now``
    (HTF frames should contain only *closed* bars).
    """

    symbol: str
    now: datetime
    exec_df: pd.DataFrame
    bias_df: pd.DataFrame
    dol_df: Optional[pd.DataFrame] = None
    partner_exec_df: Optional[pd.DataFrame] = None
    partner_symbol: Optional[str] = None

    def dol_frame(self) -> pd.DataFrame:
        return self.dol_df if self.dol_df is not None else self.bias_df


def _dedupe_pools(pools: List[LiquidityPool]) -> List[LiquidityPool]:
    seen = set()
    out: List[LiquidityPool] = []
    for p in pools:
        key = (p.type, round(p.price, 4))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


class StrategyEngine:
    def __init__(self, config: Config):
        self.config = config
        self.params = config.strategy
        self.risk = config.risk

    # -- helpers ---------------------------------------------------------

    def _enabled_killzones(self, now: datetime, symbol: str):
        kzs = active_killzones(now, symbol)
        if self.config.enabled_killzones:
            kzs = [k for k in kzs if k.name in self.config.enabled_killzones]
        return kzs

    def _pools(self, ctx: MarketContext):
        params = self.params
        pools: List[LiquidityPool] = []
        dol_df = ctx.dol_frame()
        if dol_df is not None and not dol_df.empty:
            htf_swings = find_swings(dol_df, params.htf_swing_lookback)
            pools += liquidity_pools(dol_df, htf_swings, ctx.now, params)
        exec_swings = find_swings(ctx.exec_df, params.swing_lookback)
        pools += liquidity_pools(ctx.exec_df, exec_swings, ctx.now, params)
        return _dedupe_pools(pools), exec_swings

    def _targets(self, side: Side, entry: float, risk: float) -> List[float]:
        rr1, rr2 = self.risk.rr_target, self.risk.second_target_rr
        if side is Side.LONG:
            return [entry + rr1 * risk, entry + rr2 * risk]
        return [entry - rr1 * risk, entry - rr2 * risk]

    def _midnight_open(self, exec_df: pd.DataFrame, now: datetime) -> Optional[float]:
        mid = ny_midnight(now)
        sub = exec_df[exec_df.index >= mid]
        if sub.empty or sub.index[0].date() != mid.date():
            return None
        return float(sub["open"].iloc[0])

    # -- main ------------------------------------------------------------

    def evaluate(self, ctx: MarketContext) -> Optional[Signal]:
        params = self.params
        exec_df = ctx.exec_df
        if exec_df is None or len(exec_df) < 2 * params.swing_lookback + 3:
            return None
        now = ctx.now

        # Phase 1 -- time matrix
        kzs = self._enabled_killzones(now, ctx.symbol)
        if not kzs:
            return None
        if params.require_macro and not in_macro(now):
            return None
        killzone_name = kzs[0].name

        # Phase 2 -- HTF bias -> trade side
        bias = determine_bias(ctx.bias_df, params)
        side = bias_to_side(bias)
        if side is None:
            return None

        last = len(exec_df) - 1
        atr_series = atr(exec_df, params.atr_period)
        atr_last = float(atr_series.iloc[-1])
        if atr_last <= 0:
            return None

        # Phase 5.4 -- trigger on a *fresh* FVG of `side` completing on this bar
        # (its third candle is the current bar, i.e. middle index == last - 1).
        fresh = [g for g in find_fvgs(exec_df, side, params.fvg_min_size_atr * atr_last)
                 if g.index == last - 1]
        if not fresh:
            return None
        gap = fresh[-1]
        entry = gap.entry

        pools, exec_swings = self._pools(ctx)
        if not pools:
            return None

        # Phase 5.1 -- the liquidity sweep preceding the gap (deepest matching intent)
        sweep = None
        lo_scan = max(0, last - params.sweep_lookback)
        for s_idx in range(gap.index, lo_scan - 1, -1):
            cand = detect_sweep(exec_df, pools, s_idx, params)
            if cand is None or cand.intent is not side:
                continue
            if sweep is None:
                sweep = cand
            elif side is Side.LONG and cand.extreme < sweep.extreme:
                sweep = cand
            elif side is Side.SHORT and cand.extreme > sweep.extreme:
                sweep = cand
        if sweep is None:
            return None

        # Phase 5.2 -- a Market Structure Shift between the sweep and now
        mss = detect_mss(exec_df, exec_swings, side, sweep.index, params.mss_max_bars)
        if mss is None or not (sweep.index < mss.break_index <= last):
            return None

        # Phase 5.3 -- displacement inside the breaking leg
        disp_idx = find_displacement(exec_df, atr_series, params, sweep.index, last, side)
        if disp_idx is None:
            return None

        # Phase 3 -- premium / discount filter on the entry vs the HTF dealing range
        reasons: List[str] = []
        dol_df = ctx.dol_frame()
        if params.require_premium_discount and dol_df is not None and not dol_df.empty:
            htf_swings = find_swings(dol_df, params.htf_swing_lookback)
            dr = dealing_range(dol_df, params.htf_dealing_range_lookback, htf_swings)
            if dr is not None:
                if not side_allowed(dr, side, entry):
                    return None
                zone = "discount" if dr.is_discount(entry) else "premium"
                reasons.append(f"entry in {zone} of HTF range ({dr.position(entry) * 100:.0f}%)")

        # Phase 4 -- SMT confirmation (required when a partner is configured)
        smt = None
        if ctx.partner_exec_df is not None and ctx.partner_symbol:
            smt = detect_smt(
                exec_df, ctx.partner_exec_df, side, ctx.symbol, ctx.partner_symbol,
                params.smt_swing_lookback, as_of=now,
            )
            if smt is None:
                return None
            reasons.append(f"SMT divergence vs {ctx.partner_symbol}")

        # Phase 6 -- stop at the sweep extreme, mechanical targets
        buffer = 0.1 * atr_last
        if side is Side.LONG:
            stop = sweep.extreme - buffer
            risk = entry - stop
        else:
            stop = sweep.extreme + buffer
            risk = stop - entry
        if risk <= 0:
            return None
        targets = self._targets(side, entry, risk)

        # Confluence narrative
        mo = self._midnight_open(exec_df, now)
        if mo is not None:
            if side is Side.LONG and sweep.extreme < mo:
                reasons.append("Judas swing below midnight open")
            elif side is Side.SHORT and sweep.extreme > mo:
                reasons.append("Judas swing above midnight open")
        reasons.insert(0, f"{killzone_name} killzone")
        reasons.insert(1, f"swept {sweep.pool.label} @ {sweep.pool.price:.2f}")
        reasons.append(f"MSS: close past {mss.broken_swing.type.value} @ {mss.broken_swing.price:.2f}")
        reasons.append("institutional displacement")
        reasons.append(f"FVG {gap.bottom:.2f}-{gap.top:.2f}")
        if in_macro(now):
            reasons.append("inside algo macro window")

        return Signal(
            ts=now, symbol=ctx.symbol, side=side, entry=entry, stop=stop, targets=targets,
            killzone=killzone_name, bias=bias, reasons=reasons, fvg=gap, sweep=sweep, smt=smt,
            rr=self.risk.rr_target,
        )
