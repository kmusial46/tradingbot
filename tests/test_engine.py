from datetime import date

from tradingbot.config import Config
from tradingbot.data.synthetic import SyntheticProvider
from tradingbot.timeframes import resample, closed_until
from tradingbot.timeutils import in_killzone
from tradingbot.strategy.engine import StrategyEngine, MarketContext
from tradingbot.models import Side, Bias


def _setup():
    cfg = Config.default()
    cfg.timeframes.execution = "1min"
    cfg.strategy.require_premium_discount = False  # synthetic trends; gate tested in indicators
    prov = SyntheticProvider(cfg, end_date=date(2026, 5, 22))
    return cfg, prov


def _ctx(cfg, prov, sym, partner, now, exec_full, part_full, bias_full):
    i = exec_full.index.get_loc(now)
    win = cfg.backtest.window
    exec_df = exec_full.iloc[max(0, i - win): i + 1]
    part_df = part_full[part_full.index <= now].iloc[-win:] if part_full is not None else None
    bias_df = closed_until(bias_full, now, cfg.timeframes.bias_tf).iloc[-200:]
    return MarketContext(sym, now.to_pydatetime(), exec_df, bias_df, bias_df, part_df, partner)


def test_engine_emits_valid_long_setup():
    cfg, prov = _setup()
    engine = StrategyEngine(cfg)
    exec_full = prov.get_history("NQ", "1min")
    part_full = prov.get_history("ES", "1min")
    bias_full = resample(exec_full, "1h")

    signals = []
    for i in range(50, len(exec_full)):
        now = exec_full.index[i]
        if not in_killzone(now, "NQ"):
            continue
        sig = engine.evaluate(_ctx(cfg, prov, "NQ", "ES", now, exec_full, part_full, bias_full))
        if sig:
            signals.append(sig)
        if any(s.side is Side.LONG for s in signals) and len(signals) >= 3:
            break  # enough to validate; avoid scanning the whole week

    assert signals, "expected at least one setup on the synthetic week"
    longs = [s for s in signals if s.side is Side.LONG]
    assert longs
    s = longs[0]
    assert s.killzone in {"NY Open", "NY Silver Bullet", "NY PM"}
    assert s.bias is Bias.BULLISH
    assert s.smt is not None and s.smt.partner == "ES"
    # geometry of a long: stop < entry < target
    assert s.stop < s.entry < s.targets[0]
    assert abs(s.reward_risk() - cfg.risk.rr_target) < 0.2


def test_engine_time_gate_blocks_outside_killzone():
    cfg, prov = _setup()
    engine = StrategyEngine(cfg)
    exec_full = prov.get_history("NQ", "1min")
    part_full = prov.get_history("ES", "1min")
    bias_full = resample(exec_full, "1h")

    # 12:30 NY on day0 is outside every killzone -> no setup regardless of structure.
    noon = [ts for ts in exec_full.index if ts.date() == date(2026, 5, 18) and ts.hour == 12 and ts.minute == 30][0]
    assert not in_killzone(noon, "NQ")
    sig = engine.evaluate(_ctx(cfg, prov, "NQ", "ES", noon, exec_full, part_full, bias_full))
    assert sig is None
