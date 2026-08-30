"""
Orchestration layer.

Pulls every data source together into a single ``MarketData`` bundle that the
dashboard (or a notebook) can consume. This is the one place that knows about all
the loaders, so callers just ask for ``load_market_data(days)`` and get back a
tidy object plus a record of which sources were live vs. sample.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

import config
from src.data import cache, crypto, fx, macro, preprocessing, sentiment


@dataclass
class MarketData:
    crypto_frames: dict[str, pd.DataFrame]   # symbol -> price/volume/market_cap
    crypto_prices: pd.DataFrame              # wide price matrix (columns = symbols)
    macro_prices: pd.DataFrame               # traditional-asset prices
    sentiment: pd.DataFrame                  # fear & greed
    onchain: pd.DataFrame                    # BTC on-chain metrics
    sources: dict[str, str]                  # source label per feed
    crypto_sources: dict[str, str]           # source label per individual coin
    crypto_errors: dict[str, str]            # why a coin fell back, if it did
    fx_rate: pd.Series                       # USD -> display currency, daily
    reports: list[preprocessing.CleaningReport]   # what cleaning did, per feed

    @property
    def quality(self) -> pd.DataFrame:
        """Preprocessing summary for every feed, as one table."""
        return preprocessing.quality_table(self.reports)

    @property
    def unhealthy(self) -> list[str]:
        """Datasets that failed a quality check, so the UI can name them."""
        return [r.dataset for r in self.reports if not r.is_healthy]

    @property
    def latest_fx(self) -> float:
        """Most recent USD/INR rate, for the conversion note in the UI."""
        return fx.latest_rate(self.fx_rate)

    @property
    def symbols(self) -> list[str]:
        return list(self.crypto_frames.keys())

    @property
    def simulated_symbols(self) -> list[str]:
        """Coins that fell back to generated data, so the UI can name them."""
        return [s for s, src in self.crypto_sources.items() if src != "live"]

    @property
    def failure_reasons(self) -> list[str]:
        """Distinct reasons behind any fallback, for display in the UI."""
        return list(dict.fromkeys(self.crypto_errors.values()))


def _label(source: str, keys: list[str]) -> str:
    """
    Refine a loader's source label using whether the data came off disk.

    A loader reports ``"live"`` when the value it returned originated from a
    real fetch. That is true of a cached response too — it was fetched, just not
    now — so calling both "live" leaves the interface claiming a working
    connection while a connection test fails, which is what a reader running
    offline sees. ``"cached"`` says the honest thing: real provider data, read
    from disk, not fetched this run.
    """
    if source != "live":
        return source
    return "cached" if any(cache.was_cached(k) for k in keys) else "live"


def load_market_data(
    days: int = config.DEFAULT_DAYS,
    progress: Callable[[int, int, str], None] | None = None,
) -> MarketData:
    """
    Load and assemble all feeds. Never raises for network reasons.

    ``progress(done, total, symbol)`` is passed straight through to the crypto
    loader, which is the only slow step: a keyless cold fetch spaces six calls
    six seconds apart. Without a progress signal the dashboard shows a blank
    spinner for half a minute and users conclude it has hung.
    """
    reports: list[preprocessing.CleaningReport] = []

    # Track which feeds are answered from the shipped cache rather than fetched,
    # so the interface can say which it was instead of calling both "live".
    cache.begin_load()

    frames, crypto_src, per_coin, crypto_errs = crypto.load_all_crypto(days, progress=progress)

    # Every feed is cleaned in one place and reports what it did, rather than
    # having ad-hoc dropna/astype calls scattered through the loaders. The
    # reports are surfaced on the Data & Sources tab so a reader can see exactly
    # what happened to the data before any statistic was computed from it.
    for symbol, frame in list(frames.items()):
        frames[symbol], report = preprocessing.clean_crypto_frame(frame, symbol, days)
        reports.append(report)

    prices = crypto.close_price_matrix(frames)

    macro_df, macro_src = macro.load_macro(days)

    # Crypto arrives already denominated (CoinGecko supports INR natively), but
    # Yahoo Finance quotes equities, gold and oil in dollars, so those columns
    # are converted here. The dollar index and the 10Y yield are left alone --
    # neither is a price in dollars. See src/data/fx.py for the reasoning.
    fx_series, fx_src = fx.load_usd_inr(days)
    macro_df = fx.convert_macro(macro_df, fx_series)
    macro_df, macro_report = preprocessing.clean_macro_frame(macro_df, days)
    reports.append(macro_report)

    sent_df, sent_src = sentiment.load_sentiment(days)
    sent_df, sent_report = preprocessing.clean_sentiment_frame(sent_df, days)
    reports.append(sent_report)

    onchain_df, onchain_src = sentiment.load_onchain(days)
    onchain_df, onchain_report = preprocessing.clean_onchain_frame(onchain_df, days)
    reports.append(onchain_report)

    return MarketData(
        crypto_frames=frames,
        crypto_prices=prices,
        macro_prices=macro_df,
        sentiment=sent_df,
        onchain=onchain_df,
        crypto_sources=per_coin,
        crypto_errors=crypto_errs,
        fx_rate=fx_series,
        reports=reports,
        sources={
            "Crypto (CoinGecko)": _label(crypto_src, [
                f"crypto:{cid}:{days}:{config.CURRENCY}" for cid in config.CRYPTO_ASSETS
            ]),
            "Macro (Yahoo Finance)": _label(macro_src, [f"macro:{days}"]),
            "Sentiment (Fear & Greed)": _label(sent_src, [f"fng:{days}"]),
            "On-chain (Blockchain.info)": _label(onchain_src, [f"onchain:{days}"]),
            f"FX (USD/{config.CURRENCY_LABEL})": _label(
                fx_src, [f"fx:{config.FX_TICKER}:{days}"]
            ),
        },
    )
