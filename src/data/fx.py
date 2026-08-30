"""
Foreign-exchange conversion, so the whole dashboard can be denominated in one
currency.

Crypto prices are fetched from CoinGecko directly in the target currency --
``vs_currency=inr`` is supported natively, so no conversion is applied to them
and no rounding error is introduced.

Traditional assets are a different matter. Yahoo Finance quotes the S&P 500,
gold and crude oil in US dollars, so those are converted with the daily USD/INR
rate. Two of the macro series are deliberately *not* converted:

* **US Dollar Index** is an index level, not a price in dollars.
* **US 10Y Yield** is a rate in percentage points.

Multiplying either by an exchange rate would be meaningless.

A note on method, because it is the obvious question in a viva: converting
prices to INR before differencing means the returns carry the USD/INR move as a
shared component. That is not a distortion -- it is the correct frame for an
investor whose portfolio is denominated in rupees. Their actual gain on gold is
the gold move *and* the currency move, and a diversification decision made in
INR should be measured in INR.
"""
from __future__ import annotations

import logging

import pandas as pd

import config
from src.data import cache

log = logging.getLogger(__name__)


def _fetch_usd_inr(days: int) -> pd.Series:
    """Daily USD/INR close from Yahoo Finance."""
    import yfinance as yf

    raw = yf.download(
        config.FX_TICKER,
        period=f"{days}d",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no FX data")

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index)
    series = close.sort_index().ffill().dropna()
    if series.empty:
        raise RuntimeError("FX series empty after cleaning")
    return series.rename("usd_inr")


def load_usd_inr(days: int = config.DEFAULT_DAYS) -> tuple[pd.Series, str]:
    """
    Return ``(series, source)`` for the USD/INR rate.

    Falls back to a flat series at ``config.USD_INR_FALLBACK`` so a dead FX feed
    costs accuracy rather than taking the dashboard down -- the same fail-open
    principle the other loaders follow.
    """
    if config.CURRENCY == "usd":
        # No conversion needed; a rate of 1.0 keeps every call site uniform.
        index = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")
        return pd.Series(1.0, index=index, name="usd_inr"), "n/a"

    key = f"fx:{config.FX_TICKER}:{days}"

    def _producer() -> tuple[pd.Series, str]:
        try:
            return _fetch_usd_inr(days), "live"
        except Exception as exc:  # noqa: BLE001 - fail open, never block the app
            log.warning("FX fetch failed (%s); using fallback rate", exc)
            index = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")
            return pd.Series(config.USD_INR_FALLBACK, index=index, name="usd_inr"), "sample"

    return cache.cached(key, _producer)


def latest_rate(fx: pd.Series | None) -> float:
    """The most recent USD/INR rate, or the configured fallback."""
    if fx is None or fx.empty:
        return config.USD_INR_FALLBACK if config.CURRENCY != "usd" else 1.0
    return float(fx.iloc[-1])


def convert_macro(macro_prices: pd.DataFrame, fx: pd.Series) -> pd.DataFrame:
    """
    Convert USD-quoted macro columns into the target currency.

    Only the columns listed in ``config.USD_QUOTED_ASSETS`` are touched. The
    dollar index and the 10-year yield are left exactly as they are, because
    neither is a price in dollars.
    """
    if config.CURRENCY == "usd" or macro_prices.empty:
        return macro_prices

    converted = macro_prices.copy()
    rate = fx.reindex(converted.index).ffill().bfill()
    if rate.isna().all():
        rate = pd.Series(config.USD_INR_FALLBACK, index=converted.index)

    for column in converted.columns:
        if column in config.USD_QUOTED_ASSETS:
            converted[column] = converted[column] * rate
    return converted


def format_currency(value: float, decimals: int | None = None) -> str:
    """
    Format a number in the target currency using that currency's own digit
    grouping.

    Indian grouping is not the Western three-digit pattern: it groups the last
    three digits, then pairs beyond that, so 7456772 reads as 74,56,772 rather
    than 7,456,772. Getting this wrong is immediately visible to an Indian
    reader, and pandas/format specifiers have no built-in for it.
    """
    if value is None or pd.isna(value):
        return "—"

    if decimals is None:
        magnitude = abs(value)
        decimals = 2 if magnitude < 1000 else 0

    if config.CURRENCY != "inr":
        return f"{config.CURRENCY_SYMBOL}{value:,.{decimals}f}"

    negative = value < 0
    whole = f"{abs(value):.{decimals}f}"
    fraction = ""
    if "." in whole:
        whole, fraction = whole.split(".")
        fraction = "." + fraction

    if len(whole) > 3:
        last_three = whole[-3:]
        rest = whole[:-3]
        # Pair off everything above the final three digits.
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        grouped = ",".join(parts + [last_three])
    else:
        grouped = whole

    sign = "-" if negative else ""
    return f"{sign}{config.CURRENCY_SYMBOL}{grouped}{fraction}"


def compact_currency(value: float) -> str:
    """
    Short form using Indian scale words, for axis labels and tight metric tiles.

    A crore is 10 million and a lakh is 100 thousand; an Indian reader parses
    "74.6 L" far faster than "7,456,772".
    """
    if value is None or pd.isna(value):
        return "—"
    if config.CURRENCY != "inr":
        for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
            if abs(value) >= threshold:
                return f"{config.CURRENCY_SYMBOL}{value / threshold:.1f}{suffix}"
        return f"{config.CURRENCY_SYMBOL}{value:,.0f}"

    for threshold, suffix in ((1e7, " Cr"), (1e5, " L"), (1e3, "K")):
        if abs(value) >= threshold:
            return f"{config.CURRENCY_SYMBOL}{value / threshold:,.2f}{suffix}"
    # Sub-rupee assets are real -- XRP and ADA both trade there -- so rounding to
    # whole rupees below this point would render them as a flat 0 or 1.
    decimals = 2 if abs(value) < 100 else 0
    return f"{config.CURRENCY_SYMBOL}{value:,.{decimals}f}"
