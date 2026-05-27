"""Near-live scanning loop: poll the provider, evaluate, alert on new setups.

Signals-only. Each poll fetches recent bars for every enabled instrument, builds
a context at the last closed bar, and runs the strategy engine. New setups are
de-duplicated (one alert per FVG) before being emitted to the notifier.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, Optional, Tuple

from ..config import Config
from ..data.base import DataProvider
from ..timeframes import resample, closed_until
from ..timeutils import NY, in_killzone
from ..alerts.notifier import Notifier
from ..strategy.engine import MarketContext, StrategyEngine


class LiveRunner:
    def __init__(self, config: Config, provider: DataProvider, notifier: Optional[Notifier] = None):
        self.config = config
        self.provider = provider
        self.engine = StrategyEngine(config)
        self.notifier = notifier or Notifier(config)
        self._last_alert: Dict[str, Tuple[datetime, float]] = {}

    def _fetch(self, symbol: str, timeframe: str, limit: int):
        return self.provider.get_latest(symbol, timeframe, limit=limit)

    def scan_once(self) -> int:
        cfg = self.config
        tf = cfg.timeframes
        emitted = 0
        for ins in cfg.instruments:
            if not ins.enabled:
                continue
            sym = ins.symbol
            exec_df = self._fetch(sym, tf.execution, cfg.backtest.window + 5)
            if exec_df is None or len(exec_df) < 10:
                continue
            now = exec_df.index[-1]
            if not in_killzone(now, sym):
                continue
            partner = ins.smt_partner
            partner_df = self._fetch(partner, tf.execution, cfg.backtest.window + 5) if partner else None
            bias_full = self._fetch(sym, tf.bias_tf, 300)
            dol_full = self._fetch(sym, tf.dol_tf, 300)
            bias_df = closed_until(bias_full, now, tf.bias_tf)
            dol_df = closed_until(dol_full, now, tf.dol_tf)
            ctx = MarketContext(sym, now.to_pydatetime(), exec_df, bias_df, dol_df, partner_df, partner)
            sig = self.engine.evaluate(ctx)
            if sig is None:
                continue
            prev = self._last_alert.get(sym)
            if prev and prev[0] == sig.ts and abs(prev[1] - sig.entry) < 1e-9:
                continue  # already alerted this exact setup
            self._last_alert[sym] = (sig.ts, sig.entry)
            self.notifier.emit(sig)
            emitted += 1
        return emitted

    def run(self, poll_seconds: int = 30, max_iterations: Optional[int] = None) -> None:
        self.notifier.info(
            f"Live scan started — provider={self.provider.name}, "
            f"exec={self.config.timeframes.execution}, poll={poll_seconds}s. (signals only)"
        )
        i = 0
        try:
            while max_iterations is None or i < max_iterations:
                try:
                    n = self.scan_once()
                    if n == 0:
                        self.notifier.info(f"scan @ {datetime.now(NY):%H:%M:%S} — no new setups")
                except Exception as exc:  # keep the loop alive on transient errors
                    self.notifier.info(f"scan error: {exc!r}")
                i += 1
                if max_iterations is not None and i >= max_iterations:
                    break
                time.sleep(poll_seconds)
        except KeyboardInterrupt:
            self.notifier.info("Live scan stopped.")
