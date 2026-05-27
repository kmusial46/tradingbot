from datetime import date

import pytest

from tradingbot.config import Config
from tradingbot.data.synthetic import SyntheticProvider
from tradingbot.backtest.engine import BacktestEngine
from tradingbot.backtest.report import compute_stats, trades_dataframe
from tradingbot.models import TradeOutcome


@pytest.fixture(scope="module")
def result():
    cfg = Config.default()
    cfg.timeframes.execution = "1min"
    cfg.strategy.require_premium_discount = False
    prov = SyntheticProvider(cfg, end_date=date(2026, 5, 22))
    res = BacktestEngine(cfg, prov).run(symbols=["NQ", "ES"])
    return cfg, res


def test_backtest_produces_trades(result):
    cfg, res = result
    assert res.signals > 0
    assert len(res.trades) > 0


def test_backtest_trade_geometry_and_outcomes(result):
    cfg, res = result
    for t in res.trades:
        if t.side.value == "long":
            assert t.signal.stop < t.entry < t.target
        else:
            assert t.signal.stop > t.entry > t.target
    outcomes = {t.outcome for t in res.trades}
    assert TradeOutcome.WIN in outcomes
    wins = [t for t in res.trades if t.outcome is TradeOutcome.WIN]
    assert all(abs(t.r_multiple - cfg.risk.rr_target) < 0.2 for t in wins)


def test_backtest_stats_and_equity(result):
    cfg, res = result
    stats = compute_stats(res)
    for key in ("trades", "win_rate", "total_R", "expectancy_R", "profit_factor", "max_drawdown"):
        assert key in stats
    assert res.equity_curve is not None and len(res.equity_curve) == len(
        [t for t in res.trades if t.exit_ts is not None]
    )
    assert res.ending_equity != res.starting_equity
    assert not trades_dataframe(res).empty
