"""
Tests for the deterministic offline generator.

The determinism test deliberately spawns subprocesses. Python randomises string
hashing per process, so the original `abs(hash(symbol))` seed produced identical
data within one run and different data across runs -- meaning an in-process
assertion would have passed against the buggy code and proved nothing. Only a
cross-process check can catch it, which is why CHANGES.md section 2 says the fix
was "verified identical across three separate interpreter runs".
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import config
from src.analysis import correlation
from src.data import sample_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _fingerprint_in_subprocess(symbol: str = "BTC", days: int = 365) -> str:
    """Generate a series in a fresh interpreter and return a stable digest."""
    code = textwrap.dedent(
        f"""
        import sys, hashlib
        sys.path.insert(0, r"{PROJECT_ROOT}")
        from src.data import sample_data
        s = sample_data.crypto_ohlcv("{symbol}", {days})["price"]
        print(hashlib.sha256(s.to_numpy().tobytes()).hexdigest())
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


class TestDeterminism:
    def test_identical_across_three_interpreter_runs(self):
        """
        CHANGES.md section 2. With the old hash()-derived seed these three
        digests differ, because PYTHONHASHSEED is randomised per process.
        """
        digests = {_fingerprint_in_subprocess() for _ in range(3)}
        assert len(digests) == 1, f"generator is not reproducible across runs: {digests}"

    def test_identical_within_a_process(self):
        a = sample_data.crypto_ohlcv("BTC", 365)["price"].to_numpy()
        b = sample_data.crypto_ohlcv("BTC", 365)["price"].to_numpy()
        assert np.array_equal(a, b)

    def test_different_symbols_give_different_series(self):
        btc = sample_data.crypto_ohlcv("BTC", 365)["price"].to_numpy()
        eth = sample_data.crypto_ohlcv("ETH", 365)["price"].to_numpy()
        assert not np.array_equal(btc, eth)

    def test_seed_derivation_does_not_use_builtin_hash(self):
        """Guards against the fix being reverted: the module must import zlib."""
        source = (PROJECT_ROOT / "src" / "data" / "sample_data.py").read_text(encoding="utf8")
        assert "zlib.crc32" in source
        assert "abs(hash(" not in source


class TestCorrelationStructure:
    def test_assets_are_not_independent_random_walks(self, crypto_prices, macro_prices):
        """
        CHANGES.md section 3. Independent GBM produced roughly zero correlation
        everywhere, so the correlation tab showed sampling noise rather than a
        finding. This asserts the generator's intended structure.
        """
        returns = correlation.align_returns(crypto_prices, macro_prices)
        summary = correlation.crypto_macro_summary(
            returns, ["BTC", "ETH"], list(macro_prices.columns)
        )
        assert summary.loc["BTC", "S&P 500"] > 0.15, (
            "crypto shows no risk-on loading -- the shared factor is missing"
        )

    def test_crypto_assets_correlate_strongly_with_each_other(self, crypto_prices):
        corr = crypto_prices.pct_change().dropna().corr()
        assert corr.loc["BTC", "ETH"] > 0.5

    def test_dollar_index_is_negatively_loaded(self, macro_prices):
        """A strong dollar is risk-off, so DXY should lean against equities."""
        rets = correlation._to_returns(macro_prices).dropna()
        assert rets["US Dollar Index"].corr(rets["S&P 500"]) < 0.1


class TestGeneratedShape:
    def test_price_anchored_to_the_configured_latest_value(self):
        """
        CHANGES.md section 16: paths are rescaled so the LAST value is the
        anchor. A walk left to run from a fixed start drifted to $233k.

        The profile anchors are written in US dollars because that is how these
        assets are quoted at source, so the expected value scales with the
        display currency.
        """
        expected = 95_000 * (config.USD_INR_FALLBACK if config.CURRENCY == "inr" else 1)
        price = sample_data.crypto_ohlcv("BTC", 365)["price"]
        assert price.iloc[-1] == pytest.approx(expected, rel=1e-6)

    def test_rescaling_leaves_returns_untouched(self):
        """
        Multiplying a series by a constant changes no return, so volatility,
        VaR, drawdown, correlation and every phase label are unaffected.
        """
        from src.analysis import volatility

        price = sample_data.crypto_ohlcv("BTC", 365)["price"]
        scaled = price * 3.7
        assert volatility.annualised_volatility(price) == pytest.approx(
            volatility.annualised_volatility(scaled)
        )
        assert volatility.max_drawdown(price) == pytest.approx(
            volatility.max_drawdown(scaled)
        )

    def test_volatility_is_realistic_not_textbook(self):
        """Mid-forties for BTC, not the 65% that produced a $34k-$166k range."""
        from src.analysis import volatility

        vol = volatility.annualised_volatility(sample_data.crypto_ohlcv("BTC", 365)["price"])
        assert 0.35 < vol < 0.60

    def test_market_cap_tracks_price_via_fixed_supply(self):
        """
        CHANGES.md section 7: the supply multiplier was redrawn every day, so
        market cap wandered independently of price. Supply is now fixed per
        asset, making the ratio constant.
        """
        df = sample_data.crypto_ohlcv("BTC", 365)
        implied_supply = df["market_cap"] / df["price"]
        assert implied_supply.std() / implied_supply.mean() < 1e-9

    def test_macro_prices_land_on_business_days_only(self):
        """
        CHANGES.md section 3: sample mode previously produced 365 macro rows
        against live mode's ~252, so the two modes exercised different code
        paths through the correlation alignment.
        """
        macro = sample_data.macro_prices(365)
        assert macro.index.dayofweek.max() <= 4
        assert 240 < len(macro) < 275

    def test_requested_length_is_honoured(self):
        for days in (90, 180, 365):
            assert len(sample_data.crypto_ohlcv("BTC", days)) == days

    def test_all_configured_coins_generate(self):
        for symbol in config.CRYPTO_ASSETS.values():
            df = sample_data.crypto_ohlcv(symbol, 90)
            assert not df.empty
            assert (df["price"] > 0).all()

    def test_fear_greed_stays_in_bounds(self):
        fng, _ = sample_data.fear_greed(365), None
        assert fng["fng_value"].between(0, 100).all()

    def test_onchain_metrics_are_positive(self):
        oc = sample_data.onchain_btc(365)
        assert (oc["tx_count"] > 0).all()
        assert (oc["active_addresses"] > 0).all()
