from datetime import time

from tradingbot.timeutils import (
    active_killzones, active_macro, in_killzone, ny_midnight, ny_time, normalize_symbol,
)


def test_killzone_windows():
    assert in_killzone(ny_time(2026, 5, 18, 8, 0), "NQ")        # NY Open
    assert {kz.name for kz in active_killzones(ny_time(2026, 5, 18, 8, 0), "NQ")} == {"NY Open"}
    assert in_killzone(ny_time(2026, 5, 18, 3, 0), "EURUSD")    # London
    assert in_killzone(ny_time(2026, 5, 18, 10, 30), "ES")      # Silver Bullet
    assert in_killzone(ny_time(2026, 5, 18, 15, 0), "NQ")       # NY PM


def test_killzone_symbol_filtering():
    # 03:00 is London (forex), not a futures killzone.
    assert in_killzone(ny_time(2026, 5, 18, 3, 0), "EURUSD")
    assert not in_killzone(ny_time(2026, 5, 18, 3, 0), "NQ")
    # 12:30 is outside every defined killzone.
    assert not in_killzone(ny_time(2026, 5, 18, 12, 30), "NQ")


def test_macros():
    assert active_macro(ny_time(2026, 5, 18, 9, 55)).name == "AM Macro"
    assert active_macro(ny_time(2026, 5, 18, 11, 0)).name == "Silver Bullet Macro"
    assert active_macro(ny_time(2026, 5, 18, 12, 0)) is None


def test_midnight_open_anchor():
    mid = ny_midnight(ny_time(2026, 5, 18, 8, 30))
    assert mid.hour == 0 and mid.minute == 0
    assert mid.date() == ny_time(2026, 5, 18, 8, 30).date()


def test_symbol_normalization():
    assert normalize_symbol("CME_MINI:NQ1!") == "NQ"
    assert normalize_symbol("XAUUSD") == "GOLD"
    assert normalize_symbol("MES") == "ES"
    assert normalize_symbol("EURUSD") == "EURUSD"
