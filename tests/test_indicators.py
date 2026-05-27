from tradingbot.config import StrategyParams
from tradingbot.models import DealingRange, Side, SwingType
from tradingbot.indicators.structure import find_swings, detect_mss
from tradingbot.indicators.fvg import find_fvgs
from tradingbot.indicators.liquidity import liquidity_pools, detect_sweep
from tradingbot.indicators.fibonacci import dealing_range, side_allowed
from tradingbot.indicators.smt import detect_smt

PARAMS = StrategyParams()


def test_bullish_fvg(make_frame):
    df = make_frame([(9, 10, 8, 9.5), (9.5, 14, 9.4, 13.5), (13.5, 15, 12, 14)])
    gaps = find_fvgs(df, Side.LONG)
    assert len(gaps) == 1
    g = gaps[0]
    assert g.side is Side.LONG
    assert g.bottom == 10 and g.top == 12
    assert g.entry == 12  # proximal boundary for a long


def test_bearish_fvg(make_frame):
    df = make_frame([(15, 16, 14, 14.5), (14.5, 14.6, 10, 10.5), (10.5, 12, 9, 9.5)])
    gaps = find_fvgs(df, Side.SHORT)
    assert len(gaps) == 1
    g = gaps[0]
    assert g.side is Side.SHORT
    assert g.top == 14 and g.bottom == 12
    assert g.entry == 12


def test_find_swings(make_frame):
    df = make_frame([
        (100, 105, 99, 104), (104, 108, 103, 107), (107, 112, 106, 110),
        (110, 111, 104, 105), (105, 106, 100, 101), (101, 107, 102, 106),
        (106, 114, 105, 113),
    ])
    swings = find_swings(df, 2)
    highs = [s for s in swings if s.type is SwingType.HIGH]
    lows = [s for s in swings if s.type is SwingType.LOW]
    assert any(abs(s.price - 112) < 1e-9 for s in highs)
    assert any(abs(s.price - 100) < 1e-9 for s in lows)


def test_detect_mss_long(make_frame):
    df = make_frame([
        (100, 105, 99, 104), (104, 108, 103, 107), (107, 112, 106, 110),
        (110, 111, 104, 105), (105, 106, 100, 101), (101, 107, 102, 106),
        (106, 114, 105, 113),
    ])
    swings = find_swings(df, 2)
    mss = detect_mss(df, swings, Side.LONG, after_index=4, max_bars=5)
    assert mss is not None
    assert mss.side is Side.LONG
    assert abs(mss.broken_swing.price - 112) < 1e-9  # closed past the prior swing high
    assert mss.break_close > 112


def test_detect_sweep_long(make_frame):
    df = make_frame([
        (105, 106, 104, 105), (105, 106, 103, 104), (104, 105, 101, 102),
        (102, 104, 102, 103), (103, 105, 102, 104), (104, 105, 103, 104),
        (104, 105, 100, 104),
    ])
    swings = find_swings(df, 2)
    pools = liquidity_pools(df, swings, df.index[-1].to_pydatetime(), PARAMS)
    sweep = detect_sweep(df, pools, index=6, params=PARAMS)
    assert sweep is not None
    assert sweep.intent is Side.LONG          # swept sell-side liquidity
    assert sweep.extreme == 100               # the wick low used for stop placement


def test_premium_discount_filter():
    dr = DealingRange(low=100.0, high=200.0)
    assert dr.is_discount(120) and not dr.is_premium(120)
    assert dr.is_premium(180) and not dr.is_discount(180)
    # only longs in discount, only shorts in premium
    assert side_allowed(dr, Side.LONG, 120)
    assert not side_allowed(dr, Side.LONG, 180)
    assert side_allowed(dr, Side.SHORT, 180)
    assert not side_allowed(dr, Side.SHORT, 120)


def test_dealing_range_from_frame(make_frame):
    df = make_frame([(150, 160, 100, 155), (155, 200, 150, 180), (180, 190, 140, 150)])
    dr = dealing_range(df, lookback=10)
    assert dr is not None
    assert dr.low == 100 and dr.high == 200
    assert dr.equilibrium == 150


def _smt_asset_lower_low(make_frame):
    return make_frame([
        (110, 111, 108, 109), (109, 110, 107, 108), (108, 109, 104, 105),
        (105, 107, 105, 106), (106, 108, 105, 107), (107, 109, 106, 108),
        (108, 110, 107, 109), (109, 110, 106, 107), (107, 108, 102, 103),
        (103, 105, 103, 104), (104, 106, 103, 105),
    ])


def test_smt_bullish_divergence(make_frame):
    asset = _smt_asset_lower_low(make_frame)
    partner_hl = make_frame([
        (50, 51, 48, 49), (49, 50, 47, 48), (48, 49, 44, 45),
        (45, 47, 45, 46), (46, 48, 45, 47), (47, 49, 46, 48),
        (48, 50, 47, 49), (49, 50, 46, 47), (47, 48, 45, 46),
        (46, 48, 46, 47), (47, 49, 46, 48),
    ])
    smt = detect_smt(asset, partner_hl, Side.LONG, "NQ", "ES", lookback=2)
    assert smt is not None and smt.side is Side.LONG

    # If the partner ALSO makes a lower low, there is no divergence.
    partner_ll = make_frame([
        (50, 51, 48, 49), (49, 50, 47, 48), (48, 49, 45, 46),
        (46, 48, 46, 47), (47, 49, 46, 48), (48, 50, 47, 49),
        (49, 51, 48, 50), (50, 51, 47, 48), (48, 49, 43, 44),
        (44, 46, 44, 45), (45, 47, 44, 46),
    ])
    assert detect_smt(asset, partner_ll, Side.LONG, "NQ", "ES", lookback=2) is None
