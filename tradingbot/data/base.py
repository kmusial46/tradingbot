"""DataProvider interface and a factory that builds one from config.

A provider returns normalized OHLCV frames (tz-aware NY ``DatetimeIndex``) for a
canonical symbol at a requested timeframe. Higher timeframes are derived by the
caller via resampling, so a provider only needs to serve its base timeframe(s).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd

from ..config import Config


class DataProvider(ABC):
    name: str = "base"

    @abstractmethod
    def get_history(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return normalized OHLCV bars for ``symbol`` at ``timeframe``."""

    def get_latest(self, symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
        """Most recent ``limit`` bars (used by the live poll loop)."""
        return self.get_history(symbol, timeframe, limit=limit)

    def close(self) -> None:  # pragma: no cover - optional cleanup hook
        pass


def build_provider(config: Config) -> DataProvider:
    kind = config.provider.type.lower()
    if kind == "synthetic":
        from .synthetic import SyntheticProvider

        return SyntheticProvider(config)
    if kind == "csv":
        from .csv_provider import CSVProvider

        return CSVProvider(config)
    if kind in ("tv_mcp", "tradingview", "tradingview_mcp"):
        from .tradingview_mcp import TradingViewMCPProvider

        return TradingViewMCPProvider(config)
    raise ValueError(f"unknown provider type: {config.provider.type!r}")
