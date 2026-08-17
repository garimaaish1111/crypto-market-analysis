"""
Market-sentiment and on-chain data.

* Sentiment  -> alternative.me crypto Fear & Greed index (free, no key). It is a
  composite of volatility, momentum, social media, surveys and dominance, so it
  serves as the project's social-media / sentiment signal.
* On-chain   -> blockchain.info charts API (Bitcoin transaction count & active
  addresses), representing the "blockchain transaction records" data source.

Both fall back to synthetic series on any failure.
"""
from __future__ import annotations

import logging

import pandas as pd
import requests

import config
from src.data import cache, sample_data

log = logging.getLogger(__name__)


def _fetch_fng(days: int) -> pd.DataFrame:
    resp = requests.get(
        config.FNG_URL,
        params={"limit": days, "format": "json"},
        timeout=config.REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    rows = resp.json()["data"]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
    df = df.set_index("timestamp").sort_index()
    out = pd.DataFrame(
        {
            "fng_value": pd.to_numeric(df["value"]),
            "fng_label": df["value_classification"],
        }
    )
    return out


def load_sentiment(days: int = config.DEFAULT_DAYS) -> tuple[pd.DataFrame, str]:
    key = f"fng:{days}"

    def _producer() -> tuple[pd.DataFrame, str]:
        try:
            return _fetch_fng(days), "live"
        except Exception as exc:  # noqa: BLE001
            log.warning("Fear & Greed fetch failed (%s); using sample data", exc)
            return sample_data.fear_greed(days), "sample"

    return cache.cached(key, _producer)


def _fetch_onchain_metric(chart: str, days: int) -> pd.Series:
    url = f"{config.BLOCKCHAIN_CHARTS}/{chart}"
    resp = requests.get(
        url,
        params={"timespan": f"{days}days", "format": "json"},
        timeout=config.REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    values = resp.json()["values"]
    return pd.Series(
        [p["y"] for p in values],
        index=pd.to_datetime([p["x"] for p in values], unit="s"),
    )


def load_onchain(days: int = config.DEFAULT_DAYS) -> tuple[pd.DataFrame, str]:
    key = f"onchain:{days}"

    def _producer() -> tuple[pd.DataFrame, str]:
        try:
            tx = _fetch_onchain_metric("n-transactions", days)
            addr = _fetch_onchain_metric("n-unique-addresses", days)
            df = pd.DataFrame({"tx_count": tx, "active_addresses": addr})
            df = df.resample("D").last().dropna()
            return df, "live"
        except Exception as exc:  # noqa: BLE001
            log.warning("On-chain fetch failed (%s); using sample data", exc)
            return sample_data.onchain_btc(days), "sample"

    return cache.cached(key, _producer)
