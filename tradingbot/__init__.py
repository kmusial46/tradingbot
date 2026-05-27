"""TJR / ICT-style trading bot.

A faithful implementation of the time-and-liquidity strategy described in the
project brief: killzone time windows, higher-timeframe bias and draw-on-
liquidity, premium/discount filtering, SMT divergence, and a liquidity-sweep ->
market-structure-shift -> fair-value-gap execution model.

The package is data-source agnostic. A live TradingView feed is provided via an
MCP client adapter (``tradingbot.data.tradingview_mcp``) that talks to a locally
running ``tradesdontlie/tradingview-mcp`` server. Synthetic and CSV providers are
included so the strategy and backtester can run anywhere.
"""

__version__ = "0.1.0"
