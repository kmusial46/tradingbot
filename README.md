# tradingbot — TJR / ICT time-and-liquidity bot

A faithful, testable implementation of the TJR/ICT strategy in the brief:
killzone time windows, higher-timeframe bias and draw-on-liquidity, premium/
discount filtering, SMT divergence, and a **liquidity-sweep → market-structure-
shift → fair-value-gap** execution model. It runs a **backtester** over historical
data and a **near-live signal scanner**.

> **Signals only.** The bot detects setups and prints/logs alerts (entry, stop,
> targets, size). It never places orders. Trading is risky; this is a research
> and education tool, not financial advice.

## Important: how TradingView data works here

Your MCP — [`tradesdontlie/tradingview-mcp`](https://github.com/tradesdontlie/tradingview-mcp)
— is a **local** tool. It does not call TradingView's servers; it drives your
**TradingView Desktop** app over Chrome DevTools (`localhost:9222`) and reads the
bars already on your charts. Consequences:

- The **live data path must run on your machine**, next to TradingView Desktop and
  the MCP server. A cloud session cannot reach `localhost:9222`.
- This repo therefore ships a **pluggable data layer**: a `tv_mcp` provider (an MCP
  client for that server) for real data on your machine, plus `synthetic` and
  `csv` providers so the strategy and backtester run and are validated anywhere.

## Quickstart (no TradingView needed)

```bash
pip install -r requirements.txt

# Backtest the strategy on a week of built-in synthetic data:
python -m tradingbot --config configs/synthetic.yaml backtest

# Or generate sample CSVs and backtest those:
python -m tradingbot gen-sample --out data/samples --days 5
python -m tradingbot backtest --provider csv --csv-dir data/samples \
    --exec-tf 1min --no-premium-discount --out out/trades.csv
```

The synthetic generator scripts real ICT setups (sweeps, displacement→FVG, SMT
divergence between NQ/ES) inside the killzones so the full Phase 1–6 pipeline
produces trades. It is a **fixture**, not a market simulator — the rosy win-rate
is by construction. Use it to validate the mechanics, not as performance proof.

## Using real TradingView data (run on your machine)

1. Install and launch the MCP server with TradingView Desktop on debug port 9222
   (see the `tradesdontlie/tradingview-mcp` README).
2. Install the client dependency: `pip install "mcp>=1.0"`.
3. Point the config at the server entry script (`provider.mcp_args`) and confirm
   the connection:

   ```bash
   python -m tradingbot --config configs/config.example.yaml tools   # lists MCP tools
   ```

4. Cache last week's bars to CSV, then backtest (premium/discount filter ON):

   ```bash
   python -m tradingbot --config configs/config.example.yaml \
       fetch --provider tv_mcp --out data --days 7
   python -m tradingbot --config configs/config.example.yaml \
       backtest --provider csv --csv-dir data
   ```

5. Run the near-live scanner during the session (alerts only):

   ```bash
   python -m tradingbot --config configs/config.example.yaml \
       live --provider tv_mcp --poll 30 --logfile alerts.log
   ```

> The `tv_mcp` provider's tool/argument names live at the top of
> `tradingbot/data/tradingview_mcp.py`; adjust them if your server version differs.

## How the brief maps to the code

| Phase | Concept | Where |
|------|---------|-------|
| 1 | Killzones, macros, midnight-open anchor (NY/DST-aware) | `timeutils.py` |
| 2 | HTF bias + draw-on-liquidity (PDH/PDL, EQH/EQL, BSL/SSL) | `strategy/bias.py`, `indicators/liquidity.py` |
| 3 | Premium/discount fib filter | `indicators/fibonacci.py` |
| 4 | SMT divergence across correlated pair (NQ↔ES) | `indicators/smt.py` |
| 5 | Sweep → MSS → displacement → FVG → limit entry | `indicators/{structure,displacement,fvg}.py`, `strategy/engine.py` |
| 6 | Stop at sweep extreme, 1:2/1:3 targets, move-to-BE at 1R, 1–2% risk | `strategy/engine.py`, `backtest/engine.py` |

The orchestration that requires **every** gate to align before emitting a signal
is `StrategyEngine.evaluate` in `tradingbot/strategy/engine.py`.

## Layout

```
tradingbot/
  config.py          timeutils.py        timeframes.py     models.py
  indicators/        structure, fvg, liquidity, displacement, fibonacci, smt
  strategy/          bias.py, engine.py  (Phase 1–6)
  backtest/          engine.py, report.py
  live/              runner.py           alerts/ notifier.py
  data/              base, synthetic, csv_provider, tradingview_mcp
  cli.py             # python -m tradingbot {backtest,live,fetch,gen-sample,tools}
configs/             config.example.yaml (full reference), synthetic.yaml (demo)
tests/               pytest suite for timeutils, indicators, engine, backtest
```

## Config

`configs/config.example.yaml` documents every setting (risk, detection
thresholds, timeframes, instruments, provider). CLI flags override the file:
`--provider`, `--symbols`, `--exec-tf`, `--days`, `--account`, `--risk-pct`,
`--no-premium-discount`, `--start/--end`.

Note: the synthetic demo disables the **premium/discount** gate because the
fixture trends (every pullback reads as premium of a trailing window). That gate
is exercised by the unit tests and is meant to run against genuine HTF structure
from real TradingView data.

## Tests

```bash
python -m pytest -q
```
