"""Pluggable market-data providers (synthetic, CSV, TradingView-MCP)."""

from .base import DataProvider, build_provider

__all__ = ["DataProvider", "build_provider"]
