"""Backtest metrics and human-readable reporting."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import pandas as pd

from ..models import Trade, TradeOutcome
from .engine import BacktestResult


def trades_dataframe(result: BacktestResult) -> pd.DataFrame:
    rows = []
    for t in result.trades:
        rows.append({
            "symbol": t.symbol,
            "side": t.side.value,
            "killzone": t.signal.killzone,
            "entry_ts": t.entry_ts,
            "exit_ts": t.exit_ts,
            "entry": round(t.entry, 4),
            "stop": round(t.signal.stop, 4),
            "target": round(t.target, 4),
            "exit": round(t.exit, 4) if t.exit is not None else None,
            "outcome": t.outcome.value,
            "R": round(t.r_multiple, 2),
            "pnl": round(t.pnl, 2),
            "be": t.moved_to_be,
        })
    return pd.DataFrame(rows)


def _max_drawdown(equity: pd.Series, starting: float) -> float:
    if equity is None or len(equity) == 0:
        return 0.0
    series = pd.concat([pd.Series([starting]), equity.reset_index(drop=True)])
    peak = series.cummax()
    dd = series - peak
    return float(dd.min())


def compute_stats(result: BacktestResult) -> Dict:
    trades = result.trades
    n = len(trades)
    wins = [t for t in trades if t.outcome is TradeOutcome.WIN]
    losses = [t for t in trades if t.outcome is TradeOutcome.LOSS]
    be = [t for t in trades if t.outcome is TradeOutcome.BREAKEVEN]
    total_r = sum(t.r_multiple for t in trades)
    total_pnl = sum(t.pnl for t in trades)
    gross_win = sum(t.pnl for t in wins)
    gross_loss = -sum(t.pnl for t in losses)
    stats = {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": len(be),
        "win_rate": (len(wins) / n * 100.0) if n else 0.0,
        "total_R": total_r,
        "expectancy_R": (total_r / n) if n else 0.0,
        "total_pnl": total_pnl,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0,
        "avg_win_R": (sum(t.r_multiple for t in wins) / len(wins)) if wins else 0.0,
        "avg_loss_R": (sum(t.r_multiple for t in losses) / len(losses)) if losses else 0.0,
        "max_drawdown": _max_drawdown(result.equity_curve, result.starting_equity),
        "starting_equity": result.starting_equity,
        "ending_equity": result.ending_equity,
        "signals": result.signals,
    }
    return stats


def _group_counts(trades: List[Trade], key) -> Dict[str, Dict]:
    out: Dict[str, Dict] = defaultdict(lambda: {"n": 0, "R": 0.0, "wins": 0})
    for t in trades:
        g = out[key(t)]
        g["n"] += 1
        g["R"] += t.r_multiple
        if t.outcome is TradeOutcome.WIN:
            g["wins"] += 1
    return out


def summary_text(result: BacktestResult) -> str:
    s = compute_stats(result)
    lines = []
    lines.append("=" * 60)
    lines.append("  TJR / ICT STRATEGY BACKTEST")
    lines.append("=" * 60)
    lines.append(f"  Signals taken      : {s['signals']}")
    lines.append(f"  Trades closed      : {s['trades']}")
    lines.append(f"  Win / Loss / BE    : {s['wins']} / {s['losses']} / {s['breakeven']}")
    lines.append(f"  Win rate           : {s['win_rate']:.1f}%")
    lines.append(f"  Total R            : {s['total_R']:+.2f}R")
    lines.append(f"  Expectancy         : {s['expectancy_R']:+.2f}R / trade")
    lines.append(f"  Avg win / avg loss : {s['avg_win_R']:+.2f}R / {s['avg_loss_R']:+.2f}R")
    pf = s["profit_factor"]
    lines.append(f"  Profit factor      : {'inf' if pf == float('inf') else f'{pf:.2f}'}")
    lines.append(f"  Net P&L            : {s['total_pnl']:+,.2f}")
    lines.append(f"  Equity             : {s['starting_equity']:,.0f} -> {s['ending_equity']:,.2f}")
    lines.append(f"  Max drawdown       : {s['max_drawdown']:,.2f}")

    by_sym = _group_counts(result.trades, lambda t: t.symbol)
    if by_sym:
        lines.append("-" * 60)
        lines.append("  By symbol:")
        for sym, g in sorted(by_sym.items()):
            wr = g["wins"] / g["n"] * 100.0 if g["n"] else 0.0
            lines.append(f"    {sym:6s}  trades={g['n']:3d}  win%={wr:5.1f}  R={g['R']:+.2f}")

    by_kz = _group_counts(result.trades, lambda t: t.signal.killzone)
    if by_kz:
        lines.append("  By killzone:")
        for kz, g in sorted(by_kz.items()):
            wr = g["wins"] / g["n"] * 100.0 if g["n"] else 0.0
            lines.append(f"    {kz:18s}  trades={g['n']:3d}  win%={wr:5.1f}  R={g['R']:+.2f}")
    lines.append("=" * 60)
    return "\n".join(lines)
