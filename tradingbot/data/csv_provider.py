"""CSV data provider.

Reads OHLCV CSVs from a directory. Looks for ``{SYMBOL}_{timeframe}.csv`` first
(e.g. ``NQ_5min.csv``); otherwise falls back to a 1-minute base file
(``NQ_1min.csv`` or ``NQ.csv``) and resamples. Each CSV needs a time column
(``time`` / ``ts`` / ``datetime`` / ``date``) plus open, high, low, close and an
optional volume column.

This is the bridge for real data: run ``scripts/fetch_data.py`` (TradingView MCP)
or export from TradingView to populate the directory, then backtest against it.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from ..config import Config
from ..timeframes import normalize_frame, resample
from .base import DataProvider


class CSVProvider(DataProvider):
    name = "csv"

    def __init__(self, config: Config):
        self.config = config
        self.dir = config.provider.csv_dir
        self._cache: Dict[str, pd.DataFrame] = {}

    def _load_base(self, symbol: str) -> pd.DataFrame:
        if symbol in self._cache:
            return self._cache[symbol]
        for name in (f"{symbol}_1min.csv", f"{symbol}_1m.csv", f"{symbol}.csv"):
            path = os.path.join(self.dir, name)
            if os.path.exists(path):
                df = normalize_frame(pd.read_csv(path))
                self._cache[symbol] = df
                return df
        raise FileNotFoundError(
            f"no base CSV for {symbol!r} in {self.dir!r} "
            f"(expected {symbol}_1min.csv or {symbol}.csv)"
        )

    def get_history(self, symbol, timeframe, start=None, end=None, limit=None):
        # Prefer an exact timeframe file if present.
        exact = os.path.join(self.dir, f"{symbol}_{timeframe}.csv")
        if os.path.exists(exact):
            df = normalize_frame(pd.read_csv(exact))
        else:
            base = self._load_base(symbol)
            df = base if timeframe in ("1min", "1m") else resample(base, timeframe)
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index <= end]
        if limit is not None:
            df = df.iloc[-limit:]
        return df.copy()
