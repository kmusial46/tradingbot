"""OHLC frame utilities: normalization, resampling, ATR, previous-day levels.

The base series is the execution timeframe (1min/5min). Higher timeframes are
derived by resampling so a single fetch drives the whole multi-timeframe stack.
All frames carry a tz-aware ``DatetimeIndex`` in New York time; bars are stamped
at their *open* time (``label='left', closed='left'``).
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .timeutils import NY

OHLC_COLUMNS = ["open", "high", "low", "close", "volume"]
_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean OHLCV frame: NY tz-aware sorted DatetimeIndex, float columns."""
    out = df.copy()
    # case-insensitive columns so exported CSVs ("Time", "Open", "Volume", ...) work
    out.columns = [str(c).strip().lower() for c in out.columns]
    if not isinstance(out.index, pd.DatetimeIndex):
        # try a 'ts'/'time'/'datetime' column
        for col in ("ts", "time", "datetime", "date"):
            if col in out.columns:
                out = out.set_index(pd.to_datetime(out[col]))
                out = out.drop(columns=[col])
                break
        else:
            raise ValueError("frame has no DatetimeIndex and no ts/time/datetime column")
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        idx = idx.tz_localize(NY)
    else:
        idx = idx.tz_convert(NY)
    out.index = idx
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    if "volume" not in out.columns:
        out["volume"] = 0.0
    out = out[[c for c in OHLC_COLUMNS if c in out.columns]]
    return out.astype(float)


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample an OHLCV frame to ``rule`` (e.g. '5min', '1h', '4h', 'D')."""
    out = df.resample(rule, label="left", closed="left").agg(_AGG)
    return out.dropna(subset=["open"])


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average true range (Wilder smoothing via EWM)."""
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=1).mean()


def previous_day_high_low(df: pd.DataFrame, ref_ts: datetime) -> Tuple[Optional[float], Optional[float]]:
    """Previous NY-day high/low relative to the calendar day of ``ref_ts``."""
    ref = ref_ts.astimezone(NY) if ref_ts.tzinfo else ref_ts.replace(tzinfo=NY)
    today = ref.date()
    days = df.index.date
    prior_mask = days < today
    if not prior_mask.any():
        return None, None
    prior_day = max(d for d in set(days[prior_mask]))
    day_mask = days == prior_day
    sub = df[day_mask]
    if sub.empty:
        return None, None
    return float(sub["high"].max()), float(sub["low"].min())


def day_high_low(df: pd.DataFrame, day: date) -> Tuple[Optional[float], Optional[float]]:
    sub = df[df.index.date == day]
    if sub.empty:
        return None, None
    return float(sub["high"].max()), float(sub["low"].min())


def slice_until(df: pd.DataFrame, ts: datetime) -> pd.DataFrame:
    """All bars with index <= ts (the information available 'now')."""
    return df[df.index <= ts]


def closed_until(df: pd.DataFrame, now: datetime, freq: str) -> pd.DataFrame:
    """HTF bars that have fully *closed* by ``now`` (avoids backtest lookahead).

    A bar stamped at its open time is only usable once open + tf-duration <= now.
    """
    if df.empty:
        return df
    close_times = df.index + pd.Timedelta(freq)
    return df[close_times <= now]
