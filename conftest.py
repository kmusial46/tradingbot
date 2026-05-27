import os
import sys
from datetime import datetime

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from tradingbot.timeutils import NY  # noqa: E402


@pytest.fixture
def make_frame():
    """Build an OHLCV frame from (o,h,l,c[,v]) tuples at 1-minute steps in NY."""
    import pandas as pd

    def _make(rows, start=datetime(2026, 5, 18, 9, 0), freq="1min"):
        idx = pd.date_range(start.replace(tzinfo=NY), periods=len(rows), freq=freq)
        data = {
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [r[4] if len(r) > 4 else 100.0 for r in rows],
        }
        return pd.DataFrame(data, index=idx)

    return _make
