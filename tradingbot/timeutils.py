"""New York time matrix: killzones, algorithmic macros, and the midnight anchor.

The brief specifies all times in "New York Time (EST)". In practice ICT/TJR
times mean *New York wall-clock*, which follows daylight saving. We therefore use
the ``America/New_York`` zone (EST/EDT handled automatically) rather than a fixed
-5 offset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Iterable, List, Optional, Set

try:  # prefer stdlib; fall back to dateutil if tzdata is missing
    from zoneinfo import ZoneInfo

    NY = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    from dateutil import tz as _tz

    NY = _tz.gettz("America/New_York")


# --- symbol normalization -------------------------------------------------

_ALIASES = {
    "NQ": "NQ", "NQ1!": "NQ", "MNQ": "NQ", "NASDAQ": "NQ", "US100": "NQ",
    "NDX": "NQ", "USTEC": "NQ",
    "ES": "ES", "ES1!": "ES", "MES": "ES", "SPX": "ES", "SPX500": "ES",
    "US500": "ES", "SP500": "ES",
    "GOLD": "GOLD", "XAUUSD": "GOLD", "GC": "GOLD", "GC1!": "GOLD", "MGC": "GOLD",
    "EURUSD": "EURUSD", "GBPUSD": "GBPUSD",
}


def normalize_symbol(symbol: str) -> str:
    """Map a vendor/exchange symbol to a canonical token (NQ, ES, GOLD, ...)."""
    if not symbol:
        return symbol
    token = symbol.split(":")[-1].upper()  # drop "CME_MINI:" style prefixes
    token = "".join(ch for ch in token if ch.isalnum() or ch == "!")
    return _ALIASES.get(token, token)


# --- killzones ------------------------------------------------------------

@dataclass(frozen=True)
class Killzone:
    name: str
    start: time
    end: time
    assets: frozenset  # canonical symbols this killzone primarily applies to

    def contains_time(self, t: time) -> bool:
        # All defined killzones lie within a single day (no midnight wrap).
        return self.start <= t < self.end

    def applies_to(self, symbol: Optional[str]) -> bool:
        if symbol is None:
            return True
        return normalize_symbol(symbol) in self.assets


KILLZONES: List[Killzone] = [
    Killzone("London Open", time(2, 0), time(5, 0), frozenset({"EURUSD", "GBPUSD"})),
    Killzone("NY Open", time(7, 0), time(10, 0), frozenset({"ES", "NQ", "GOLD"})),
    Killzone("NY Silver Bullet", time(10, 0), time(11, 0), frozenset({"ES", "NQ"})),
    Killzone("NY PM", time(14, 0), time(16, 0), frozenset({"ES", "NQ"})),
]


# --- algorithmic macros (20-minute injection windows) ---------------------

@dataclass(frozen=True)
class Macro:
    name: str
    start: time
    end: time

    def contains_time(self, t: time) -> bool:
        return self.start <= t < self.end


MACROS: List[Macro] = [
    Macro("AM Macro", time(9, 50), time(10, 10)),
    Macro("Silver Bullet Macro", time(10, 50), time(11, 10)),
]


# --- helpers --------------------------------------------------------------

def to_ny(ts: datetime) -> datetime:
    """Return ``ts`` as a tz-aware datetime in New York time."""
    if ts.tzinfo is None:
        # Treat naive timestamps as already NY local.
        return ts.replace(tzinfo=NY)
    return ts.astimezone(NY)


def active_killzones(ts: datetime, symbol: Optional[str] = None) -> List[Killzone]:
    """Killzones active at ``ts`` (optionally filtered to those covering ``symbol``)."""
    t = to_ny(ts).time()
    return [kz for kz in KILLZONES if kz.contains_time(t) and kz.applies_to(symbol)]


def in_killzone(ts: datetime, symbol: Optional[str] = None) -> bool:
    return bool(active_killzones(ts, symbol))


def active_macro(ts: datetime) -> Optional[Macro]:
    t = to_ny(ts).time()
    for m in MACROS:
        if m.contains_time(t):
            return m
    return None


def in_macro(ts: datetime) -> bool:
    return active_macro(ts) is not None


def ny_midnight(ts: datetime) -> datetime:
    """The 00:00 NY anchor (true midnight open) for the NY calendar day of ``ts``."""
    n = to_ny(ts)
    return n.replace(hour=0, minute=0, second=0, microsecond=0)


def session_date(ts: datetime) -> date:
    """The NY calendar date of ``ts``."""
    return to_ny(ts).date()


def ny_time(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    """Construct a tz-aware NY datetime (convenience for tests / config)."""
    return datetime(year, month, day, hour, minute, tzinfo=NY)
