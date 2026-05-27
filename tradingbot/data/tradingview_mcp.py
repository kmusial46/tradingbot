"""TradingView data provider via the local ``tradesdontlie/tradingview-mcp`` server.

That MCP server is a *local* tool: it drives your running TradingView Desktop app
over the Chrome DevTools protocol and exposes tools like ``chart_set_symbol``,
``chart_set_timeframe``, ``data_get_ohlcv`` and ``chart_scroll_to_date``. This
provider is an MCP **client** that spawns the server over stdio and reads bars
from it.

Run this where TradingView Desktop is running (your machine), not in the cloud
sandbox. Requires the optional ``mcp`` package:  ``pip install "mcp>=1.0"``.

Tool and argument names are centralized in the constants below; if your version
of the server differs, adjust them in one place. Use ``health_check()`` / the
``tradingbot tools`` CLI command to list what your server actually exposes.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import Config
from ..timeframes import normalize_frame
from ..timeutils import NY
from .base import DataProvider

# --- server tool / argument names (adjust to match your server version) ---
TOOL_SET_SYMBOL = "chart_set_symbol"
TOOL_SET_TIMEFRAME = "chart_set_timeframe"
TOOL_GET_OHLCV = "data_get_ohlcv"
TOOL_SCROLL_TO_DATE = "chart_scroll_to_date"
TOOL_HEALTH = "tv_health_check"

ARG_SYMBOL = "symbol"
ARG_TIMEFRAME = "timeframe"
ARG_DATE = "date"

# canonical timeframe string -> TradingView timeframe token
_TF_MAP = {
    "1min": "1", "1m": "1",
    "5min": "5", "5m": "5",
    "15min": "15", "15m": "15",
    "30min": "30", "30m": "30",
    "1h": "60", "60min": "60",
    "4h": "240", "240min": "240",
    "D": "1D", "1d": "1D", "1day": "1D",
}


def _to_tv_timeframe(tf: str) -> str:
    return _TF_MAP.get(tf, tf)


def _extract_payload(result: Any) -> Any:
    """Pull a JSON payload out of an MCP tool result (structured or text)."""
    structured = getattr(result, "structuredContent", None)
    if structured:
        return structured
    content = getattr(result, "content", None)
    if not content:
        return None
    texts = []
    for item in content:
        text = getattr(item, "text", None)
        if text is not None:
            texts.append(text)
    blob = "\n".join(texts).strip()
    if not blob:
        return None
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return {"raw": blob}


def _parse_bars(payload: Any) -> List[Dict[str, float]]:
    """Normalize the many possible OHLCV shapes into a list of bar dicts."""
    if payload is None:
        return []
    # unwrap common containers
    if isinstance(payload, dict):
        for key in ("bars", "ohlcv", "data", "candles", "values"):
            if key in payload and isinstance(payload[key], list):
                payload = payload[key]
                break
        else:
            # columnar form: {"t":[...],"o":[...],...}
            if "o" in payload and "c" in payload:
                t = payload.get("t") or payload.get("time") or []
                return [
                    {"time": t[i], "open": payload["o"][i], "high": payload["h"][i],
                     "low": payload["l"][i], "close": payload["c"][i],
                     "volume": (payload.get("v") or [0] * len(t))[i]}
                    for i in range(len(t))
                ]
            return []
    bars: List[Dict[str, float]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        ts = row.get("time") or row.get("t") or row.get("timestamp") or row.get("datetime") or row.get("date")
        bars.append({
            "time": ts,
            "open": float(row.get("open", row.get("o"))),
            "high": float(row.get("high", row.get("h"))),
            "low": float(row.get("low", row.get("l"))),
            "close": float(row.get("close", row.get("c"))),
            "volume": float(row.get("volume", row.get("v", 0)) or 0),
        })
    return bars


def _bars_to_frame(bars: List[Dict[str, float]]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(bars)
    ts = pd.to_datetime(df["time"], utc=True, errors="coerce")
    if ts.isna().all():  # epoch seconds/millis
        unit = "ms" if df["time"].astype("int64").max() > 1e12 else "s"
        ts = pd.to_datetime(df["time"], unit=unit, utc=True)
    df.index = ts.dt.tz_convert(NY)
    df = df.drop(columns=["time"]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


class TradingViewMCPProvider(DataProvider):
    name = "tv_mcp"

    def __init__(self, config: Config):
        self.config = config
        self.command = config.provider.mcp_command
        self.args = list(config.provider.mcp_args)
        self.bars_per_call = config.provider.mcp_bars_per_call

    def _tv_symbol(self, symbol: str) -> str:
        ins = self.config.instrument(symbol)
        if ins and ins.tv_symbol:
            return ins.tv_symbol
        return symbol

    # -- public ----------------------------------------------------------

    def get_history(self, symbol, timeframe, start=None, end=None, limit=None):
        max_pages = 1
        if start is not None:
            # rough page budget to walk back to `start`
            max_pages = 60
        df = asyncio.run(self._afetch(self._tv_symbol(symbol), _to_tv_timeframe(timeframe), start, max_pages))
        df = normalize_frame(df) if not df.empty else df
        if start is not None and not df.empty:
            df = df[df.index >= start]
        if end is not None and not df.empty:
            df = df[df.index <= end]
        if limit is not None and not df.empty:
            df = df.iloc[-limit:]
        return df.copy()

    def get_latest(self, symbol, timeframe, limit=300):
        return self.get_history(symbol, timeframe, limit=limit)

    def health_check(self) -> Dict[str, Any]:
        return asyncio.run(self._ahealth())

    # -- async internals -------------------------------------------------

    def _session(self):
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The TradingView-MCP provider needs the 'mcp' package. "
                "Install it with:  pip install \"mcp>=1.0\""
            ) from exc

        params = StdioServerParameters(command=self.command, args=self.args)
        return stdio_client(params), ClientSession

    async def _ahealth(self) -> Dict[str, Any]:
        ctx, ClientSession = self._session()
        async with ctx as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = [t.name for t in tools.tools]
                info: Dict[str, Any] = {"tools": names, "count": len(names)}
                if TOOL_HEALTH in names:
                    info["health"] = _extract_payload(await session.call_tool(TOOL_HEALTH, {}))
                return info

    async def _afetch(self, tv_symbol: str, tv_tf: str, start: Optional[datetime], max_pages: int) -> pd.DataFrame:
        ctx, ClientSession = self._session()
        async with ctx as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await session.call_tool(TOOL_SET_SYMBOL, {ARG_SYMBOL: tv_symbol})
                await session.call_tool(TOOL_SET_TIMEFRAME, {ARG_TIMEFRAME: tv_tf})

                frames: List[pd.DataFrame] = []
                earliest: Optional[pd.Timestamp] = None
                for _ in range(max(1, max_pages)):
                    payload = _extract_payload(await session.call_tool(TOOL_GET_OHLCV, {"format": "full"}))
                    page = _bars_to_frame(_parse_bars(payload))
                    if page.empty:
                        break
                    frames.append(page)
                    page_earliest = page.index.min()
                    if earliest is not None and page_earliest >= earliest:
                        break  # no progress paging back
                    earliest = page_earliest
                    if start is None or earliest <= pd.Timestamp(start).tz_convert(NY):
                        break
                    # page further back in history
                    await session.call_tool(
                        TOOL_SCROLL_TO_DATE, {ARG_DATE: earliest.date().isoformat()}
                    )

                if not frames:
                    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
                out = pd.concat(frames)
                return out[~out.index.duplicated(keep="last")].sort_index()
