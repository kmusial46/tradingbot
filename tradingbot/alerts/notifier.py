"""Render and emit trade-setup alerts. This bot never places orders."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from ..config import Config
from ..models import Signal


def format_signal(sig: Signal, config: Optional[Config] = None) -> str:
    arrow = "LONG ▲" if sig.side.value == "long" else "SHORT ▼"
    lines = [
        "",
        "┌" + "─" * 56,
        f"│  {arrow}  {sig.symbol}   [{sig.killzone}]",
        f"│  {sig.ts:%Y-%m-%d %H:%M %Z}   bias={sig.bias.value}",
        "│" + "─" * 56,
        f"│  Entry (FVG)   : {sig.entry:.4f}",
        f"│  Stop          : {sig.stop:.4f}   (risk {sig.risk_per_unit:.4f}/unit)",
        f"│  Target 1      : {sig.targets[0]:.4f}   ({sig.reward_risk():.1f}R)",
    ]
    if len(sig.targets) > 1:
        lines.append(f"│  Target 2      : {sig.targets[1]:.4f}   ({sig.reward_risk(sig.targets[1]):.1f}R)")
    if config is not None:
        risk_amt = config.risk.account_size * config.risk.risk_pct / 100.0
        size = risk_amt / sig.risk_per_unit if sig.risk_per_unit > 0 else 0.0
        lines.append(f"│  Size (≈{config.risk.risk_pct:.0f}% risk): {size:.2f} units  (${risk_amt:,.0f})")
    lines.append("│" + "─" * 56)
    lines.append("│  Confluence:")
    for r in sig.reasons:
        lines.append(f"│    • {r}")
    lines.append("│  Management: move stop to breakeven at +1R.")
    lines.append("└" + "─" * 56)
    return "\n".join(lines)


class Notifier:
    """Emits alerts to the console and, optionally, appends to a log file."""

    def __init__(self, config: Optional[Config] = None, logfile: Optional[str] = None):
        self.config = config
        self.logfile = logfile

    def emit(self, signal: Signal) -> None:
        text = format_signal(signal, self.config)
        print(text, flush=True)
        if self.logfile:
            with open(self.logfile, "a") as fh:
                fh.write(text + "\n")

    def info(self, message: str) -> None:
        line = f"[{datetime.now():%H:%M:%S}] {message}"
        print(line, flush=True)
        if self.logfile:
            with open(self.logfile, "a") as fh:
                fh.write(line + "\n")
