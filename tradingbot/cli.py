"""Command-line interface.

    python -m tradingbot backtest   [--provider synthetic|csv|tv_mcp] [options]
    python -m tradingbot live       [--provider ...] [--poll 30]
    python -m tradingbot fetch      --out data [--days 5]      # cache history to CSV
    python -m tradingbot gen-sample --out data/samples         # synthetic CSVs
    python -m tradingbot tools                                 # list TV-MCP tools
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from typing import List, Optional

from .config import Config
from .timeutils import NY
from .data.base import build_provider


def _load_config(args) -> Config:
    cfg = Config.load(args.config) if getattr(args, "config", None) else Config.default()
    if getattr(args, "provider", None):
        cfg.provider.type = args.provider
    if getattr(args, "exec_tf", None):
        cfg.timeframes.execution = args.exec_tf
    if getattr(args, "days", None):
        cfg.provider.synthetic_days = args.days
    if getattr(args, "account", None):
        cfg.risk.account_size = args.account
    if getattr(args, "risk_pct", None):
        cfg.risk.risk_pct = args.risk_pct
    if getattr(args, "no_premium_discount", False):
        cfg.strategy.require_premium_discount = False
    if getattr(args, "csv_dir", None):
        cfg.provider.csv_dir = args.csv_dir
    return cfg


def _symbols(args, cfg: Config) -> List[str]:
    if getattr(args, "symbols", None):
        return [s.strip().upper() for s in args.symbols.split(",")]
    return [i.symbol for i in cfg.instruments if i.enabled]


def _save_csv(df, path: str) -> None:
    out = df.copy()
    out.insert(0, "time", out.index.strftime("%Y-%m-%d %H:%M:%S%z"))
    out.to_csv(path, index=False)


# --- commands -------------------------------------------------------------

def cmd_backtest(args) -> int:
    from .backtest.engine import BacktestEngine
    from .backtest.report import summary_text, trades_dataframe

    cfg = _load_config(args)
    provider = build_provider(cfg)
    symbols = _symbols(args, cfg)

    start = end = None
    if args.start:
        start = datetime.fromisoformat(args.start).replace(tzinfo=NY)
    if args.end:
        end = datetime.fromisoformat(args.end).replace(tzinfo=NY)

    engine = BacktestEngine(cfg, provider)
    result = engine.run(symbols=symbols, start=start, end=end)
    print(summary_text(result))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        trades_dataframe(result).to_csv(args.out, index=False)
        print(f"\nTrade log written to {args.out}")
        if result.equity_curve is not None and len(result.equity_curve):
            eq_path = args.out.replace(".csv", "") + "_equity.csv"
            result.equity_curve.to_csv(eq_path)
            print(f"Equity curve written to {eq_path}")
    return 0


def cmd_live(args) -> int:
    from .alerts.notifier import Notifier
    from .live.runner import LiveRunner

    cfg = _load_config(args)
    provider = build_provider(cfg)
    notifier = Notifier(cfg, logfile=args.logfile)
    runner = LiveRunner(cfg, provider, notifier)
    runner.run(poll_seconds=args.poll, max_iterations=args.iterations)
    return 0


def cmd_fetch(args) -> int:
    cfg = _load_config(args)
    provider = build_provider(cfg)
    symbols = _symbols(args, cfg)
    os.makedirs(args.out, exist_ok=True)
    start = (datetime.now(NY) - timedelta(days=args.days)) if args.days else None

    if getattr(args, "timeframes", None):
        tfs = [t.strip() for t in args.timeframes.split(",")]
    else:
        tf = cfg.timeframes
        tfs = []
        for t in [tf.execution, tf.bias_tf, tf.dol_tf, *tf.htf]:
            if t not in tfs:
                tfs.append(t)

    failures = 0
    for sym in symbols:
        for t in tfs:
            try:
                df = provider.get_history(sym, t, start=start)
            except Exception as exc:
                failures += 1
                print(f"{sym} {t}: FETCH FAILED — {exc}")
                continue
            if df.empty:
                failures += 1
                print(f"{sym} {t}: 0 bars (nothing returned). Try `mcp-debug` to inspect the server.")
                continue
            path = os.path.join(args.out, f"{sym}_{t}.csv")
            _save_csv(df, path)
            print(f"{sym} {t}: {len(df)} bars -> {path}")
    return 1 if failures else 0


def cmd_mcp_debug(args) -> int:
    import json

    cfg = _load_config(args)
    cfg.provider.type = "tv_mcp"
    provider = build_provider(cfg)
    symbols = _symbols(args, cfg)
    sym = symbols[0]
    print(f"Probing TradingView MCP for {sym} ...\n")
    try:
        info = provider.diagnose(sym)
    except Exception as exc:
        print(f"mcp-debug failed: {exc}")
        return 1
    print(json.dumps(info, indent=2, default=str))
    return 0


def cmd_gen_sample(args) -> int:
    from .data.synthetic import generate_dataset

    cfg = _load_config(args)
    symbols = _symbols(args, cfg)
    os.makedirs(args.out, exist_ok=True)
    frames = generate_dataset(symbols, days=args.days, seed=cfg.provider.synthetic_seed)
    for sym, df in frames.items():
        path = os.path.join(args.out, f"{sym}_1min.csv")
        _save_csv(df, path)
        print(f"{sym}: {len(df)} bars -> {path}")
    return 0


def cmd_tools(args) -> int:
    cfg = _load_config(args)
    cfg.provider.type = "tv_mcp"
    provider = build_provider(cfg)
    try:
        info = provider.health_check()
    except Exception as exc:
        print(f"Could not reach the TradingView MCP server: {exc}")
        print(
            "Make sure TradingView Desktop is running with --remote-debugging-port=9222, "
            "that 'mcp' is installed (pip install \"mcp>=1.0\"), and that provider.mcp_args "
            "in your config points at the server's entry script."
        )
        return 1
    print(f"TradingView MCP exposes {info.get('count', 0)} tools:")
    for name in info.get("tools", []):
        print(f"  - {name}")
    if "health" in info:
        print("health:", info["health"])
    return 0


# --- parser ---------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tradingbot", description="TJR/ICT trading bot")
    p.add_argument("--config", help="path to YAML config")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--provider", choices=["synthetic", "csv", "tv_mcp"], help="data provider")
        sp.add_argument("--symbols", help="comma-separated symbols (default: all enabled)")
        sp.add_argument("--exec-tf", dest="exec_tf", help="execution timeframe (1min, 5min, ...)")
        sp.add_argument("--csv-dir", dest="csv_dir", help="directory for the csv provider")

    bt = sub.add_parser("backtest", help="run a backtest")
    common(bt)
    bt.add_argument("--days", type=int, help="synthetic: number of weekdays to generate")
    bt.add_argument("--start", help="ISO start (csv/tv_mcp)")
    bt.add_argument("--end", help="ISO end (csv/tv_mcp)")
    bt.add_argument("--account", type=float, help="account size")
    bt.add_argument("--risk-pct", dest="risk_pct", type=float, help="risk %% per trade")
    bt.add_argument("--no-premium-discount", action="store_true",
                    help="disable the premium/discount filter (use for trending synthetic data)")
    bt.add_argument("--out", help="write trade log CSV to this path")
    bt.set_defaults(func=cmd_backtest)

    lv = sub.add_parser("live", help="run the near-live signal scanner (signals only)")
    common(lv)
    lv.add_argument("--poll", type=int, default=30, help="seconds between scans")
    lv.add_argument("--iterations", type=int, help="stop after N scans (default: run forever)")
    lv.add_argument("--logfile", help="append alerts to this file")
    lv.set_defaults(func=cmd_live)

    fe = sub.add_parser("fetch", help="cache provider history to CSV")
    common(fe)
    fe.add_argument("--out", default="data", help="output directory")
    fe.add_argument("--days", type=int, default=7, help="days of history to fetch")
    fe.add_argument("--timeframes", help="comma-separated timeframes to fetch (default: from config)")
    fe.set_defaults(func=cmd_fetch)

    gs = sub.add_parser("gen-sample", help="generate synthetic sample CSVs")
    common(gs)
    gs.add_argument("--out", default="data/samples", help="output directory")
    gs.add_argument("--days", type=int, default=5, help="number of weekdays")
    gs.set_defaults(func=cmd_gen_sample)

    tl = sub.add_parser("tools", help="list tools exposed by the TradingView MCP server")
    tl.set_defaults(func=cmd_tools)

    md = sub.add_parser("mcp-debug", help="dump raw TradingView MCP responses for one symbol")
    common(md)
    md.set_defaults(func=cmd_mcp_debug)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
