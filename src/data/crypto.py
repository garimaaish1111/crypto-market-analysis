"""
Crypto price / volume / market-cap ingestion via the CoinGecko public API.

The free tier is aggressively rate limited — roughly 5 to 15 calls per minute
depending on load — and this project asks for six coins in a row. Firing them
back to back reliably earns a 429 on the last two or three, which then fall back
to simulated data while the first few stay live. That produces the worst possible
outcome: a risk table and correlation matrix silently mixing real and generated
assets.

Three things prevent that here. Requests are throttled to one every few seconds,
429 responses are retried with exponential backoff and honour ``Retry-After``,
and the source of every individual coin is reported back so the dashboard can
name exactly which assets are simulated rather than saying "mixed" and leaving
the reader to guess.
"""
from __future__ import annotations

import logging
import time

import pandas as pd
import requests

import config
from src.data import cache, sample_data

log = logging.getLogger(__name__)

_last_request_at = 0.0


def _throttle() -> None:
    """Space out live calls. Cache hits never reach this, so warm runs are instant."""
    global _last_request_at
    elapsed = time.time() - _last_request_at
    if elapsed < config.COINGECKO_MIN_INTERVAL:
        time.sleep(config.COINGECKO_MIN_INTERVAL - elapsed)
    _last_request_at = time.time()


def _headers() -> dict[str, str]:
    headers = {"accept": "application/json"}
    if config.API_KEY:
        # Demo and Pro keys use different header names and different hosts.
        # Sending a Pro header to the public host is silently ignored, which is
        # why an apparently valid key can still get rate limited.
        key_header = "x-cg-pro-api-key" if config.COINGECKO_PLAN == "pro" else "x-cg-demo-api-key"
        headers[key_header] = config.API_KEY
    return headers


def _get_with_retry(url: str, params: dict) -> requests.Response:
    """GET with backoff on 429. Raises if every attempt is exhausted."""
    delay = config.COINGECKO_BACKOFF
    last_error: Exception | None = None

    for attempt in range(1, config.COINGECKO_MAX_ATTEMPTS + 1):
        _throttle()
        try:
            resp = requests.get(url, params=params, headers=_headers(), timeout=config.REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(delay)
            delay *= 2
            continue

        if resp.status_code == 429:
            # Honour the server's own guidance when it gives any.
            wait = float(resp.headers.get("Retry-After", delay))
            wait = min(wait, config.COINGECKO_MAX_WAIT)
            log.warning(
                "CoinGecko rate limited (attempt %d/%d); waiting %.0fs",
                attempt, config.COINGECKO_MAX_ATTEMPTS, wait,
            )
            last_error = RuntimeError("429 Too Many Requests")
            time.sleep(wait)
            delay *= 2
            continue

        resp.raise_for_status()
        return resp

    raise last_error or RuntimeError("CoinGecko request failed")


def _fetch_market_chart(coin_id: str, days: int) -> pd.DataFrame:
    """Call CoinGecko and reshape prices/volumes/market-caps into a daily frame."""
    url = f"{config.coingecko_base()}/coins/{coin_id}/market_chart"
    payload = _get_with_retry(url, {"vs_currency": "usd", "days": days}).json()

    def _series(field: str, name: str) -> pd.Series:
        arr = payload[field]
        return pd.Series(
            [v for _, v in arr],
            index=pd.to_datetime([t for t, _ in arr], unit="ms"),
            name=name,
        )

    df = pd.concat(
        [
            _series("prices", "price"),
            _series("total_volumes", "volume"),
            _series("market_caps", "market_cap"),
        ],
        axis=1,
    )
    # Collapse intraday points to one row per day.
    return df.resample("D").last().dropna()


def load_crypto(symbol: str, coin_id: str, days: int = config.DEFAULT_DAYS) -> tuple[pd.DataFrame, str]:
    """
    Return ``(dataframe, source)`` for one coin.

    ``source`` is one of:

    ``"live"``       fetched from CoinGecko
    ``"simulated"``  generated deliberately, because CRYPTO_MODE is "simulated"
    ``"sample"``     generated as an unplanned fallback after a live call failed

    The last two produce identical data but mean different things, and the
    dashboard treats them differently: a chosen mode is stated calmly, an
    unexpected failure is flagged loudly.
    """
    if config.CRYPTO_MODE != "live":
        # No network call, no exception to catch, nothing to log. The generator
        # is deterministic and fast, so there is nothing worth caching either.
        return sample_data.crypto_ohlcv(symbol, days), "simulated"

    key = f"crypto:{coin_id}:{days}"

    def _producer() -> tuple[pd.DataFrame, str]:
        try:
            return _fetch_market_chart(coin_id, days), "live"
        except Exception as exc:  # noqa: BLE001 - deliberately broad, we always recover
            log.warning("CoinGecko fetch failed for %s (%s); using sample data", coin_id, exc)
            return sample_data.crypto_ohlcv(symbol, days), "sample"

    return cache.cached(key, _producer)


def load_all_crypto(
    days: int = config.DEFAULT_DAYS,
) -> tuple[dict[str, pd.DataFrame], str, dict[str, str]]:
    """
    Load every configured coin.

    Returns the frames, an overall source label, and a per-symbol source map so
    the dashboard can name which individual coins fell back rather than reporting
    an undifferentiated "mixed".

    In simulated mode this loop never touches the network, so it returns
    immediately rather than spending the throttle interval on each coin.
    """
    frames: dict[str, pd.DataFrame] = {}
    per_symbol: dict[str, str] = {}

    for coin_id, symbol in config.CRYPTO_ASSETS.items():
        df, src = load_crypto(symbol, coin_id, days)
        frames[symbol] = df
        per_symbol[symbol] = src

    sources = set(per_symbol.values())
    if sources == {"live"}:
        overall = "live"
    elif sources == {"simulated"}:
        overall = "simulated"
    elif "live" in sources:
        overall = "mixed"
    else:
        overall = "sample"
    return frames, overall, per_symbol


def close_price_matrix(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine per-coin frames into a single price matrix (columns = symbols)."""
    prices = {sym: df["price"] for sym, df in frames.items()}
    return pd.DataFrame(prices).sort_index().ffill().dropna(how="all")
