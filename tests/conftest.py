"""
Shared fixtures.

Every test runs against the deterministic generator rather than a live feed, so
the suite is offline, fast, and gives the same answer on every machine. That is
only sound because ``sample_data`` is genuinely reproducible — which is itself
asserted in ``test_sample_data.py``.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from src.data import sample_data  # noqa: E402

config.CRYPTO_MODE = "simulated"


@pytest.fixture(scope="session")
def btc_price() -> pd.Series:
    """365 days of deterministic BTC prices."""
    return sample_data.crypto_ohlcv("BTC", 365)["price"]


@pytest.fixture(scope="session")
def crypto_prices() -> pd.DataFrame:
    """Wide price matrix for three coins."""
    return pd.DataFrame(
        {s: sample_data.crypto_ohlcv(s, 365)["price"] for s in ("BTC", "ETH", "SOL")}
    )


@pytest.fixture(scope="session")
def macro_prices() -> pd.DataFrame:
    return sample_data.macro_prices(365)


@pytest.fixture
def rising() -> pd.Series:
    """A strictly increasing series — no losing day anywhere in it."""
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    return pd.Series(np.linspace(100.0, 300.0, 120), index=idx)


@pytest.fixture
def falling() -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    return pd.Series(np.linspace(300.0, 100.0, 120), index=idx)


@pytest.fixture
def flat() -> pd.Series:
    """Completely unchanging — genuinely neutral momentum."""
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    return pd.Series(np.full(120, 100.0), index=idx)
