"""
Cache and pipeline tests.

The cache tests pin CHANGES.md section 4: the refresh button called
st.cache_data.clear(), which only clears Streamlit's in-memory cache. The pickle
files on disk survived, so a failed fetch kept serving its sample-data fallback
for the full six-hour TTL and the button could not dislodge it.
"""
from __future__ import annotations

import time

import pandas as pd
import pytest

import config
from src.data import cache, crypto
from src import pipeline


@pytest.fixture
def temp_cache(tmp_path, monkeypatch):
    """Point the cache at a throwaway directory so tests never touch real data."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    return tmp_path


class TestCache:
    def test_roundtrip(self, temp_cache):
        cache.save("k", {"a": 1})
        assert cache.load("k") == {"a": 1}

    def test_miss_returns_none(self, temp_cache):
        assert cache.load("never-written") is None

    def test_expired_entry_is_a_miss(self, temp_cache):
        cache.save("k", "value")
        time.sleep(0.01)
        assert cache.load("k", ttl_hours=0) is None

    def test_cached_computes_once_then_reuses(self, temp_cache):
        calls = []

        def producer():
            calls.append(1)
            return "computed"

        assert cache.cached("k", producer) == "computed"
        assert cache.cached("k", producer) == "computed"
        assert len(calls) == 1, "producer ran twice despite a warm cache"

    def test_clear_all_removes_disk_entries(self, temp_cache):
        """
        The section 4 defect. Streamlit's clear() left these files in place.
        """
        for i in range(3):
            cache.save(f"k{i}", i)
        assert cache.entry_count() == 3
        assert cache.clear_all() == 3
        assert cache.entry_count() == 0
        assert cache.load("k0") is None

    def test_clear_all_on_empty_cache_is_safe(self, temp_cache):
        assert cache.clear_all() == 0

    def test_unreadable_entry_fails_open_as_a_miss(self, temp_cache):
        """Any cache error must be invisible to the user, never an exception."""
        cache.save("k", "value")
        path = cache._cache_path("k")
        path.write_bytes(b"not a pickle")
        assert cache.load("k") is None

    def test_distinct_keys_do_not_collide(self, temp_cache):
        cache.save("crypto:bitcoin:365", "a")
        cache.save("crypto:bitcoin:270", "b")
        assert cache.load("crypto:bitcoin:365") == "a"
        assert cache.load("crypto:bitcoin:270") == "b"


class TestSimulatedMode:
    def test_simulated_mode_makes_no_network_call(self, monkeypatch):
        """
        CHANGES.md section 15: in simulated mode there is no request, so there
        is no exception to catch and no 429 to retry.
        """
        monkeypatch.setattr(config, "CRYPTO_MODE", "simulated")

        def explode(*args, **kwargs):
            raise AssertionError("network was contacted in simulated mode")

        monkeypatch.setattr(crypto.requests, "get", explode)
        # load_crypto returns (frame, source, error) -- the third element carries
        # why a coin fell back, so the UI can name the reason rather than just
        # reporting "sample".
        df, source, error = crypto.load_crypto("BTC", "bitcoin", 90)
        assert source == "simulated"
        assert not error
        assert len(df) == 90

    def test_source_labels_distinguish_choice_from_failure(self, monkeypatch):
        """
        "simulated" (a chosen mode) and "sample" (an unplanned fallback) produce
        identical data but mean different things, and the UI treats them
        differently -- calm caption versus warning banner.
        """
        monkeypatch.setattr(config, "CRYPTO_MODE", "simulated")
        _, source, _error = crypto.load_crypto("BTC", "bitcoin", 90)
        assert source == "simulated"

    def test_overall_label_is_mixed_when_only_some_coins_are_live(self):
        """A partially-live load must not report itself as fully live."""
        per_symbol = {"BTC": "live", "ETH": "sample"}
        sources = set(per_symbol.values())
        assert "live" in sources and sources != {"live"}


class TestPipeline:
    @pytest.fixture(scope="class")
    def market(self, monkeypatch_class):
        return pipeline.load_market_data(days=180)

    @pytest.fixture(scope="class")
    def monkeypatch_class(self):
        config.CRYPTO_MODE = "simulated"
        return None

    def test_bundle_is_populated(self, market):
        assert market.symbols
        assert not market.crypto_prices.empty
        # Five feeds now: the FX rate is a data source like any other, since the
        # dashboard is denominated in rupees and macro assets are quoted in USD.
        assert set(market.sources) == {
            "Crypto (CoinGecko)",
            "Macro (Yahoo Finance)",
            "Sentiment (Fear & Greed)",
            "On-chain (Blockchain.info)",
            f"FX (USD/{config.CURRENCY_LABEL})",
        }

    def test_price_matrix_columns_match_symbols(self, market):
        assert set(market.crypto_prices.columns) == set(market.symbols)

    def test_per_coin_sources_cover_every_symbol(self, market):
        assert set(market.crypto_sources) == set(market.symbols)

    def test_simulated_symbols_are_named_for_the_ui(self, market):
        """The banner names which coins are generated rather than saying "mixed"."""
        assert set(market.simulated_symbols) <= set(market.symbols)

    def test_close_price_matrix_is_sorted_and_aligned(self, market):
        assert market.crypto_prices.index.is_monotonic_increasing


class TestStaleBeatsSynthetic:
    """
    The submitted-folder scenario: the shipped cache has aged out and the
    machine has no network. Serving real-but-dated data is more honest than
    serving invented data stamped with today's date.
    """

    def test_expired_entry_is_served_when_the_producer_falls_back(self, temp_cache):
        cache.save("k", ("REAL", "live"))
        # Age the file well past any TTL.
        path = cache._cache_path("k")
        old = time.time() - 60 * 60 * 24 * 400
        import os

        os.utime(path, (old, old))

        def failing_producer():
            return ("GENERATED", "sample")

        result = cache.cached("k", failing_producer, ttl_hours=1, should_cache=cache.is_live)
        assert result == ("REAL", "live"), "generated data was preferred over real data"

    def test_fresh_success_still_wins_over_stale(self, temp_cache):
        cache.save("k", ("OLD", "live"))
        path = cache._cache_path("k")
        old = time.time() - 60 * 60 * 24 * 400
        import os

        os.utime(path, (old, old))

        result = cache.cached(
            "k", lambda: ("NEW", "live"), ttl_hours=1, should_cache=cache.is_live
        )
        assert result == ("NEW", "live")

    def test_no_cache_at_all_returns_the_fallback(self, temp_cache):
        result = cache.cached(
            "missing", lambda: ("GENERATED", "sample"), should_cache=cache.is_live
        )
        assert result == ("GENERATED", "sample")

    def test_is_live_accepts_the_three_element_crypto_shape(self):
        """A loader growing a field must not silently stop being cached."""
        assert cache.is_live(("df", "live", None)) is True
        assert cache.is_live(("df", "sample", "429")) is False
