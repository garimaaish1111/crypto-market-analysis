"""
Per-feed connection diagnostics.

This is what the dashboard's "Test connection" buttons call — one real,
cheap request against each provider, timed, with a human-readable reason
for any failure. It never raises: a dead feed comes back as ``ok=False``
with the reason in ``message``, so a broken network never turns a
diagnostics panel into a stack trace.

Each check is deliberately independent of the caching and retry machinery
in ``crypto.py`` / ``macro.py`` / ``sentiment.py`` — those exist to make the
*application* resilient, which would hide exactly the thing this module is
asked to reveal: what the network is doing right now, this second.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import pandas as pd
import requests

import config
from src.data import crypto

log = logging.getLogger(__name__)

# The four feeds, in the order the dashboard shows them.
FEEDS: list[str] = ["coingecko", "yahoo", "fng", "blockchain"]

_PROVIDER: dict[str, str] = {
    "coingecko": "CoinGecko",
    "yahoo": "Yahoo Finance",
    "fng": "alternative.me",
    "blockchain": "Blockchain.info",
}

_DATASET: dict[str, str] = {
    "coingecko": "Crypto prices",
    "yahoo": "Traditional assets",
    "fng": "Market sentiment",
    "blockchain": "Bitcoin on-chain",
}

# Yahoo Finance's own chart endpoint, hit directly with `requests` rather than
# through yfinance. That makes the exact URL reportable and the check genuinely
# cheap (one ticker, five days) instead of paying yfinance's session/threading
# overhead for a single probe.
_YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_YAHOO_PROBE_TICKER = next(iter(config.MACRO_ASSETS))
_YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; crypto-market-analysis/health-check)"}


@dataclass
class FeedHealth:
    key: str            # stable id: "coingecko" | "yahoo" | "fng" | "blockchain"
    provider: str       # "CoinGecko", "Yahoo Finance", "alternative.me", "Blockchain.info"
    dataset: str        # "Crypto prices", "Traditional assets", "Market sentiment", "Bitcoin on-chain"
    ok: bool            # did a real request succeed
    latency_ms: float   # round-trip time; float("nan") if it never returned
    message: str        # one short human line, e.g. "200 OK, 365 points" / "connect timeout after 20s"
    endpoint: str        # the exact URL that was tested
    checked_at: pd.Timestamp


def _network_failure_message(host: str, exc: Exception) -> str:
    if isinstance(exc, requests.Timeout):
        return f"connect timeout after {config.REQUEST_TIMEOUT}s"
    if isinstance(exc, requests.ConnectionError):
        return f"could not reach {host} (no connection, DNS failure, or a firewall in the way)"
    return f"request failed: {exc}"


def _check_coingecko() -> tuple[bool, str, str]:
    endpoint = f"{config.coingecko_base()}/ping"
    try:
        resp = requests.get(endpoint, headers=crypto._headers(), timeout=config.REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        return False, _network_failure_message("api.coingecko.com", exc), endpoint

    if resp.status_code == 200:
        return True, "200 OK", endpoint
    if resp.status_code == 429:
        return False, "429 rate limited (the keyless tier allows roughly 10-30 calls/min)", endpoint
    if resp.status_code == 401:
        return False, "401 — API key rejected", endpoint
    if resp.status_code == 403:
        return False, "403 — refused, often a proxy/VPN/firewall rather than CoinGecko itself", endpoint
    return False, f"HTTP {resp.status_code}", endpoint


def _check_yahoo() -> tuple[bool, str, str]:
    endpoint = _YAHOO_CHART_URL.format(ticker=_YAHOO_PROBE_TICKER)
    try:
        resp = requests.get(
            endpoint,
            params={"range": "5d", "interval": "1d"},
            headers=_YAHOO_HEADERS,
            timeout=config.REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, _network_failure_message("query1.finance.yahoo.com", exc), endpoint

    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}", endpoint
    try:
        result = resp.json()["chart"]["result"][0]
        n = len(result["timestamp"])
    except Exception:
        return False, "200 OK but the response wasn't the expected chart JSON", endpoint
    return True, f"200 OK, {n} points ({_YAHOO_PROBE_TICKER})", endpoint


def _check_fng() -> tuple[bool, str, str]:
    endpoint = config.FNG_URL
    try:
        resp = requests.get(
            endpoint, params={"limit": 1, "format": "json"}, timeout=config.REQUEST_TIMEOUT
        )
    except requests.RequestException as exc:
        return False, _network_failure_message("api.alternative.me", exc), endpoint

    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}", endpoint
    try:
        n = len(resp.json()["data"])
    except Exception:
        return False, "200 OK but the response wasn't the expected JSON shape", endpoint
    return True, f"200 OK, {n} point(s)", endpoint


def _check_blockchain() -> tuple[bool, str, str]:
    endpoint = f"{config.BLOCKCHAIN_CHARTS}/n-transactions"
    try:
        resp = requests.get(
            endpoint, params={"timespan": "2days", "format": "json"}, timeout=config.REQUEST_TIMEOUT
        )
    except requests.RequestException as exc:
        return False, _network_failure_message("api.blockchain.info", exc), endpoint

    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}", endpoint
    try:
        n = len(resp.json()["values"])
    except Exception:
        return False, "200 OK but the response wasn't the expected JSON shape", endpoint
    return True, f"200 OK, {n} point(s)", endpoint


_CHECKERS = {
    "coingecko": _check_coingecko,
    "yahoo": _check_yahoo,
    "fng": _check_fng,
    "blockchain": _check_blockchain,
}


def check_feed(key: str) -> FeedHealth:
    """Make ONE real request against that provider and time it. Never raises."""
    provider = _PROVIDER.get(key, "unknown")
    dataset = _DATASET.get(key, "unknown")
    checker = _CHECKERS.get(key)

    if checker is None:
        return FeedHealth(
            key=key,
            provider=provider,
            dataset=dataset,
            ok=False,
            latency_ms=float("nan"),
            message=f"unknown feed key {key!r} (expected one of {FEEDS})",
            endpoint="",
            checked_at=pd.Timestamp.now(),
        )

    start = time.perf_counter()
    try:
        ok, message, endpoint = checker()
    except Exception as exc:  # noqa: BLE001 - a health probe must never raise
        ok, message, endpoint = False, f"unexpected error: {exc}", ""
        log.warning("Health check for %s raised unexpectedly", key, exc_info=True)
    latency_ms = (time.perf_counter() - start) * 1000.0

    return FeedHealth(
        key=key,
        provider=provider,
        dataset=dataset,
        ok=ok,
        latency_ms=latency_ms,
        message=message,
        endpoint=endpoint,
        checked_at=pd.Timestamp.now(),
    )


def check_all() -> list[FeedHealth]:
    """Every feed in FEEDS order. Never raises."""
    return [check_feed(key) for key in FEEDS]
