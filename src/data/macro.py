"""
Traditional / macroeconomic asset prices via Yahoo Finance (yfinance).

Provides the equities, commodities, dollar-index and rates series used for the
digital-vs-traditional correlation analysis. Falls back to synthetic prices if
yfinance is unavailable or returns nothing.
"""
from __future__ import annotations

import logging

import pandas as pd

import config
from src.data import cache, sample_data

log = logging.getLogger(__name__)


def _fetch_macro(days: int) -> pd.DataFrame:
    import yfinance as yf  # imported lazily so the app starts without it

    tickers = list(config.MACRO_ASSETS.keys())

    # Requesting exactly `days` produces a ragged frame: Gold, Crude Oil and
    # the US Dollar Index currently resume their continuous futures front on
    # 2025-06-20, about 65 days after the S&P 500 and the 10Y yield
    # (2025-03-18). ffill cannot backfill a *leading* gap, so those three
    # columns would start life as NaN and the quality table would flag macro
    # as "Check" on every run even though nothing is actually broken. Pulling
    # extra history and trimming to the first date every column has real data
    # fixes that regardless of which tickers happen to lag, or by how much, so
    # it keeps working if the gap moves or a sixth ticker is added later.
    buffer_days = max(120, days // 3)
    raw = yf.download(
        tickers,
        period=f"{days + buffer_days}d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data")

    close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    close = close.rename(columns=config.MACRO_ASSETS)
    close.index = pd.to_datetime(close.index)
    close = close.sort_index().ffill()

    # After forward-fill, the only NaNs left are the leading ones before each
    # column's own first observation. Find the first row where every column
    # already has a real value, and start there.
    complete = close.dropna(how="any")
    if complete.empty:
        # No date has every ticker populated (e.g. one feed is entirely
        # down). Better to hand back what's live, forward-filled, than to
        # raise and lose the whole macro feed to sample data.
        return close.dropna(how="all").tail(days)

    return close.loc[complete.index[0]:].tail(days)


def load_macro(days: int = config.DEFAULT_DAYS) -> tuple[pd.DataFrame, str]:
    key = f"macro:{days}"

    def _producer() -> tuple[pd.DataFrame, str]:
        try:
            return _fetch_macro(days), "live"
        except Exception as exc:  # noqa: BLE001
            log.warning("Macro fetch failed (%s); using sample data", exc)
            return sample_data.macro_prices(days), "sample"

    # Successes only: a cached failure would pin the dashboard to generated
    # data for the full TTL even after the network recovered.
    return cache.cached(key, _producer, should_cache=cache.is_live)
