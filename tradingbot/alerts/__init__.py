"""Signal alerting (console / file). Signals-only: no orders are ever placed."""

from .notifier import Notifier, format_signal

__all__ = ["Notifier", "format_signal"]
