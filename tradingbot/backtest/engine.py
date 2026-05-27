"""Event-driven backtester.

Replays the execution-timeframe bars for each instrument through
``StrategyEngine.evaluate``. A signal places a limit order at the FVG boundary;
subsequent bars fill it (price trading back into the gap), then the trade is
managed to its stop / target with a move-to-breakeven at +1R. At most one
live-or-pending trade per instrument (configurable), which also dedupes the
multiple gaps a single displacement leg can leave.

Backtest conventions (kept deliberately pessimistic):
  * a limit fills at its exact price when the bar's range touches it;
  * if a single bar touches both stop and target, the stop is assumed first.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from ..config import Config
from ..data.base import DataProvider
from ..models import Side, Signal, Trade, TradeOutcome
from ..timeframes import closed_until
from ..timeutils import in_killzone
from ..strategy.engine import MarketContext, StrategyEngine


@dataclass
class _Pending:
    signal: Signal
    side: Side
    entry: float
    stop: float
    target: float
    r_unit: float
    risk_amount: float
    size: float
    age: int = 0


@dataclass
class BacktestResult:
    config: Config
    trades: List[Trade] = field(default_factory=list)
    signals: int = 0
    bars: int = 0
    starting_equity: float = 0.0
    equity_curve: Optional[pd.Series] = None

    @property
    def ending_equity(self) -> float:
        if self.equity_curve is not None and len(self.equity_curve):
            return float(self.equity_curve.iloc[-1])
        return self.starting_equity


class BacktestEngine:
    def __init__(self, config: Config, provider: DataProvider):
        self.config = config
        self.provider = provider
        self.engine = StrategyEngine(config)

    # -- public ----------------------------------------------------------

    def run(self, symbols: Optional[List[str]] = None, start=None, end=None) -> BacktestResult:
        cfg = self.config
        if symbols is None:
            symbols = [i.symbol for i in cfg.instruments if i.enabled]

        all_trades: List[Trade] = []
        total_signals = 0
        total_bars = 0
        for sym in symbols:
            trades, signals, bars = self._run_symbol(sym, start, end)
            all_trades.extend(trades)
            total_signals += signals
            total_bars += bars

        result = BacktestResult(
            config=cfg, trades=all_trades, signals=total_signals, bars=total_bars,
            starting_equity=cfg.risk.account_size,
        )
        result.equity_curve = self._equity_curve(all_trades, cfg.risk.account_size)
        return result

    # -- per-symbol pass -------------------------------------------------

    def _run_symbol(self, symbol: str, start, end):
        cfg = self.config
        tf = cfg.timeframes
        exec_full = self.provider.get_history(symbol, tf.execution, start=start, end=end)
        if exec_full is None or len(exec_full) <= cfg.backtest.warmup_bars:
            return [], 0, 0

        ins = cfg.instrument(symbol)
        partner = ins.smt_partner if ins else None
        partner_full = (
            self.provider.get_history(partner, tf.execution, start=start, end=end)
            if partner else None
        )
        bias_full = self.provider.get_history(symbol, tf.bias_tf, start=start, end=end)
        dol_full = self.provider.get_history(symbol, tf.dol_tf, start=start, end=end)

        risk_amount = cfg.risk.account_size * cfg.risk.risk_pct / 100.0
        idx = exec_full.index

        trades: List[Trade] = []
        signals = 0
        pending: Optional[_Pending] = None
        open_trade: Optional[Trade] = None

        for i in range(cfg.backtest.warmup_bars, len(exec_full)):
            now = idx[i]
            bar = exec_full.iloc[i]

            # 1) manage an open trade against this bar
            if open_trade is not None and self._manage(open_trade, bar, now):
                trades.append(open_trade)
                open_trade = None

            # 2) try to fill a resting limit
            if open_trade is None and pending is not None:
                open_trade = self._try_fill(pending, bar, now, symbol)
                if open_trade is not None:
                    pending = None
                    if self._manage(open_trade, bar, now):  # same-bar resolution
                        trades.append(open_trade)
                        open_trade = None
                else:
                    pending.age += 1
                    if pending.age > cfg.backtest.pending_expiry_bars:
                        pending = None

            # 3) look for a new setup when flat and inside a killzone
            flat = open_trade is None and (pending is None or not cfg.backtest.one_trade_per_symbol)
            if flat and in_killzone(now, symbol):
                ctx = self._context(symbol, partner, now, i, exec_full, partner_full, bias_full, dol_full)
                sig = self.engine.evaluate(ctx)
                if sig is not None:
                    signals += 1
                    r_unit = sig.risk_per_unit
                    size = risk_amount / r_unit if r_unit > 0 else 0.0
                    pending = _Pending(
                        signal=sig, side=sig.side, entry=sig.entry, stop=sig.stop,
                        target=sig.targets[0], r_unit=r_unit, risk_amount=risk_amount, size=size,
                    )

        return trades, signals, len(exec_full) - cfg.backtest.warmup_bars

    # -- context ---------------------------------------------------------

    def _context(self, symbol, partner, now, i, exec_full, partner_full, bias_full, dol_full) -> MarketContext:
        tf = self.config.timeframes
        win = self.config.backtest.window
        exec_df = exec_full.iloc[max(0, i - win): i + 1]
        partner_df = None
        if partner_full is not None:
            partner_df = partner_full[partner_full.index <= now].iloc[-win:]
        bias_df = closed_until(bias_full, now, tf.bias_tf).iloc[-200:]
        dol_df = closed_until(dol_full, now, tf.dol_tf).iloc[-200:]
        return MarketContext(symbol, now.to_pydatetime(), exec_df, bias_df, dol_df, partner_df, partner)

    # -- fills & management ---------------------------------------------

    def _try_fill(self, p: _Pending, bar: pd.Series, now, symbol: str) -> Optional[Trade]:
        low, high = float(bar["low"]), float(bar["high"])
        hit = (p.side is Side.LONG and low <= p.entry) or (p.side is Side.SHORT and high >= p.entry)
        if not hit:
            return None
        return Trade(
            signal=p.signal, symbol=symbol, side=p.side, entry_ts=now.to_pydatetime(),
            entry=p.entry, stop=p.stop, target=p.target, size=p.size, risk_amount=p.risk_amount,
        )

    def _manage(self, t: Trade, bar: pd.Series, now) -> bool:
        """Update an open trade against ``bar``; return True if it closed."""
        low, high = float(bar["low"]), float(bar["high"])
        r_unit = t.signal.risk_per_unit
        be_at = self.config.risk.move_be_at_r

        if t.side is Side.LONG:
            if not t.moved_to_be and high >= t.entry + be_at * r_unit:
                t.stop = t.entry
                t.moved_to_be = True
            if low <= t.stop:
                return self._close(t, t.stop, now)
            if high >= t.target:
                return self._close(t, t.target, now)
        else:
            if not t.moved_to_be and low <= t.entry - be_at * r_unit:
                t.stop = t.entry
                t.moved_to_be = True
            if high >= t.stop:
                return self._close(t, t.stop, now)
            if low <= t.target:
                return self._close(t, t.target, now)
        return False

    def _close(self, t: Trade, price: float, now) -> bool:
        t.exit = price
        t.exit_ts = now.to_pydatetime()
        direction = 1.0 if t.side is Side.LONG else -1.0
        t.pnl = (price - t.entry) * t.size * direction
        t.r_multiple = t.pnl / t.risk_amount if t.risk_amount > 0 else 0.0
        if t.r_multiple > 0.01:
            t.outcome = TradeOutcome.WIN
        elif t.r_multiple < -0.01:
            t.outcome = TradeOutcome.LOSS
        else:
            t.outcome = TradeOutcome.BREAKEVEN
        return True

    # -- equity ----------------------------------------------------------

    @staticmethod
    def _equity_curve(trades: List[Trade], starting: float) -> pd.Series:
        closed = [t for t in trades if t.exit_ts is not None]
        closed.sort(key=lambda t: t.exit_ts)
        equity = starting
        ts_list, eq_list = [], []
        for t in closed:
            equity += t.pnl
            ts_list.append(t.exit_ts)
            eq_list.append(equity)
        return pd.Series(eq_list, index=pd.DatetimeIndex(ts_list), name="equity")
