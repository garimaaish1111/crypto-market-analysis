"""
Orchestration layer.

Pulls every data source together into a single ``MarketData`` bundle that the
dashboard (or a notebook) can consume. This is the one place that knows about all
the loaders, so callers just ask for ``load_market_data(days)`` and get back a
tidy object plus a record of which sources were live vs. sample.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import config
from src.data import crypto, macro, sentiment


@dataclass
class MarketData:
    crypto_frames: dict[str, pd.DataFrame]   # symbol -> price/volume/market_cap
    crypto_prices: pd.DataFrame              # wide price matrix (columns = symbols)
    macro_prices: pd.DataFrame               # traditional-asset prices
    sentiment: pd.DataFrame                  # fear & greed
    onchain: pd.DataFrame                    # BTC on-chain metrics
    sources: dict[str, str]                  # source label per feed
    crypto_sources: dict[str, str]           # source label per individual coin

    @property
    def symbols(self) -> list[str]:
        return list(self.crypto_frames.keys())

    @property
    def simulated_symbols(self) -> list[str]:
        """Coins that fell back to generated data, so the UI can name them."""
        return [s for s, src in self.crypto_sources.items() if src != "live"]


def load_market_data(days: int = config.DEFAULT_DAYS) -> MarketData:
    """Load and assemble all feeds. Never raises for network reasons."""
    frames, crypto_src, per_coin = crypto.load_all_crypto(days)
    prices = crypto.close_price_matrix(frames)
    macro_df, macro_src = macro.load_macro(days)
    sent_df, sent_src = sentiment.load_sentiment(days)
    onchain_df, onchain_src = sentiment.load_onchain(days)

    return MarketData(
        crypto_frames=frames,
        crypto_prices=prices,
        macro_prices=macro_df,
        sentiment=sent_df,
        onchain=onchain_df,
        crypto_sources=per_coin,
        sources={
            "Crypto (CoinGecko)": crypto_src,
            "Macro (Yahoo Finance)": macro_src,
            "Sentiment (Fear & Greed)": sent_src,
            "On-chain (Blockchain.info)": onchain_src,
        },
    )
