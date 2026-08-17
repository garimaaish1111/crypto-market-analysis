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
    raw = yf.download(
        tickers,
        period=f"{days}d",
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
    return close.sort_index().ffill().dropna(how="all")


def load_macro(days: int = config.DEFAULT_DAYS) -> tuple[pd.DataFrame, str]:
    key = f"macro:{days}"

    def _producer() -> tuple[pd.DataFrame, str]:
        try:
            return _fetch_macro(days), "live"
        except Exception as exc:  # noqa: BLE001
            log.warning("Macro fetch failed (%s); using sample data", exc)
            return sample_data.macro_prices(days), "sample"

    return cache.cached(key, _producer)
