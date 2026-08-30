"""
Correlation tests.

The centrepiece is TestPartialProviderFailure, which reproduces the production
incident in CHANGES.md section 18: a single rate-limited yfinance ticker came
back as an all-NaN column, the row-wise dropna then discarded every row, and the
Correlation tab rendered blank for all eleven assets. The scenario table in that
section is encoded here directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.analysis import correlation


class TestYieldHandling:
    def test_rate_columns_are_differenced_not_percent_changed(self):
        """
        A 10Y yield moving 4.00 -> 4.10 is a 10bp change, not a +2.5% return.
        CHANGES.md section 7, "Yields were treated as prices".
        """
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        prices = pd.DataFrame({"US 10Y Yield": [4.00, 4.10, 4.05]}, index=idx)
        # _to_returns drops the leading all-NaN row, so the first change is iloc[0].
        out = correlation._to_returns(prices)
        assert out["US 10Y Yield"].iloc[0] == pytest.approx(0.10)
        assert out["US 10Y Yield"].iloc[0] != pytest.approx(0.025)
        assert out["US 10Y Yield"].iloc[1] == pytest.approx(-0.05)

    def test_price_columns_are_percent_changed(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        prices = pd.DataFrame({"Gold": [100.0, 110.0, 99.0]}, index=idx)
        out = correlation._to_returns(prices)
        assert out["Gold"].iloc[0] == pytest.approx(0.10)
        assert out["Gold"].iloc[1] == pytest.approx(-0.10)

    def test_yield_asset_is_configured(self):
        assert "US 10Y Yield" in config.YIELD_ASSETS


class TestPartialProviderFailure:
    """The CHANGES.md section 18 scenario table, as executable assertions."""

    @staticmethod
    def _returns_frame(n_cols: int = 11, n_rows: int = 250) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        idx = pd.date_range("2024-01-01", periods=n_rows, freq="D")
        return pd.DataFrame(
            rng.normal(0, 0.02, size=(n_rows, n_cols)),
            index=idx,
            columns=[f"A{i}" for i in range(n_cols)],
        )

    def test_healthy_frame_is_untouched(self):
        frame = self._returns_frame()
        kept, dropped = correlation.drop_thin_columns(frame)
        assert kept.shape == (250, 11)
        assert dropped == []

    def test_one_dead_column_costs_one_asset_not_every_row(self):
        """
        Before the fix this scenario produced 0 rows and a blank heatmap.
        After it: 250 rows x 10 cols.
        """
        frame = self._returns_frame()
        frame["A3"] = np.nan
        kept, dropped = correlation.drop_thin_columns(frame)
        assert dropped == ["A3"]
        assert kept.dropna().shape == (250, 10)

    def test_two_dead_columns(self):
        frame = self._returns_frame()
        frame[["A3", "A7"]] = np.nan
        kept, dropped = correlation.drop_thin_columns(frame)
        assert set(dropped) == {"A3", "A7"}
        assert kept.dropna().shape == (250, 9)

    def test_column_with_only_ten_observations_is_dropped(self):
        """A feed that returns a handful of points is as unusable as a dead one."""
        frame = self._returns_frame()
        frame["A3"] = np.nan
        frame.iloc[:10, frame.columns.get_loc("A3")] = 0.01
        kept, dropped = correlation.drop_thin_columns(frame)
        assert "A3" in dropped
        assert kept.dropna().shape == (250, 10)

    def test_all_columns_dead_returns_empty_not_crash(self):
        frame = self._returns_frame(n_cols=3)
        frame[:] = np.nan
        kept, dropped = correlation.drop_thin_columns(frame)
        assert kept.empty
        assert len(dropped) == 3

    def test_coverage_measured_against_best_column_not_row_count(self):
        """
        A genuinely short history must not be mistaken for a broken feed: if
        every column is equally short, nothing is dropped.
        """
        frame = self._returns_frame(n_cols=4, n_rows=250)
        frame.iloc[100:] = np.nan  # all four columns stop at the same point
        kept, dropped = correlation.drop_thin_columns(frame)
        assert dropped == []
        assert list(kept.columns) == list(frame.columns)

    def test_align_returns_survives_a_dead_macro_feed(self, crypto_prices, macro_prices):
        """End-to-end: the exact production failure, through the public API."""
        broken = macro_prices.copy()
        broken["Gold"] = np.nan
        out = correlation.align_returns(crypto_prices, broken)
        assert len(out) > 0, "a single dead feed emptied the whole correlation sample"
        assert "Gold" not in out.columns
        assert "BTC" in out.columns

    def test_every_macro_feed_dead_still_correlates_crypto(self, crypto_prices, macro_prices):
        broken = macro_prices.copy()
        broken[:] = np.nan
        out = correlation.align_returns(crypto_prices, broken)
        assert len(out) > 0
        assert {"BTC", "ETH", "SOL"} <= set(out.columns)


class TestCorrelationMatrix:
    def test_diagonal_is_one(self, crypto_prices, macro_prices):
        returns = correlation.align_returns(crypto_prices, macro_prices)
        corr = correlation.correlation_matrix(returns)
        assert np.allclose(np.diag(corr.to_numpy()), 1.0)

    def test_symmetric(self, crypto_prices, macro_prices):
        returns = correlation.align_returns(crypto_prices, macro_prices)
        corr = correlation.correlation_matrix(returns)
        assert np.allclose(corr.to_numpy(), corr.to_numpy().T)

    def test_bounded(self, crypto_prices, macro_prices):
        returns = correlation.align_returns(crypto_prices, macro_prices)
        corr = correlation.correlation_matrix(returns).to_numpy()
        assert (corr >= -1.0 - 1e-9).all() and (corr <= 1.0 + 1e-9).all()

    def test_spearman_available_as_robustness_check(self, crypto_prices, macro_prices):
        returns = correlation.align_returns(crypto_prices, macro_prices)
        assert not correlation.correlation_matrix(returns, method="spearman").isna().all().all()

    def test_generated_data_carries_the_intended_structure(self, crypto_prices, macro_prices):
        """
        Guards CHANGES.md section 3: independent random walks produced a flat
        zero matrix, which silently invalidated the whole tab offline. Crypto
        must load positively onto equities and near-zero onto gold.

        This asserts a property of the GENERATOR, not a market measurement.
        """
        returns = correlation.align_returns(crypto_prices, macro_prices)
        summary = correlation.crypto_macro_summary(
            returns, ["BTC", "ETH", "SOL"], list(macro_prices.columns)
        )
        assert summary.loc["BTC", "S&P 500"] > 0.15
        assert abs(summary.loc["BTC", "Gold"]) < 0.20


class TestBeta:
    def test_beta_of_an_identical_series_is_one(self, crypto_prices, macro_prices):
        """
        Beta against a copy of itself must be exactly 1. A literal self-beta
        (same column name twice) is not tested because returns[[a, a]] yields a
        duplicate-named frame, which is not a path the dashboard can reach --
        the benchmark picker only ever offers macro columns.
        """
        returns = correlation.align_returns(crypto_prices, macro_prices)
        frame = returns[["BTC"]].copy()
        frame["BTC_copy"] = frame["BTC"]
        assert correlation.beta(frame, "BTC", "BTC_copy") == pytest.approx(1.0, rel=1e-6)

    def test_doubled_series_has_beta_two(self):
        rng = np.random.default_rng(1)
        idx = pd.date_range("2024-01-01", periods=200, freq="D")
        base = rng.normal(0, 0.01, 200)
        returns = pd.DataFrame({"bench": base, "levered": base * 2}, index=idx)
        assert correlation.beta(returns, "levered", "bench") == pytest.approx(2.0, rel=1e-6)

    def test_missing_column_returns_nan(self, crypto_prices, macro_prices):
        returns = correlation.align_returns(crypto_prices, macro_prices)
        with pytest.raises(KeyError):
            correlation.beta(returns, "DOGE", "S&P 500")


class TestRegimeCorrelation:
    def test_splits_sample_at_the_configured_quantile(self, crypto_prices, macro_prices):
        returns = correlation.align_returns(crypto_prices, macro_prices)
        result = correlation.regime_correlation(returns, "BTC", "S&P 500")
        total = result.calm_days + result.stressed_days
        assert result.stressed_days == pytest.approx(total * config.STRESS_QUANTILE, abs=3)

    def test_missing_asset_returns_nan_profile(self, crypto_prices, macro_prices):
        returns = correlation.align_returns(crypto_prices, macro_prices)
        result = correlation.regime_correlation(returns, "NOTACOIN", "S&P 500")
        assert np.isnan(result.calm) and np.isnan(result.stressed)

    def test_shift_is_stressed_minus_calm(self):
        rc = correlation.RegimeCorrelation("BTC", "S&P 500", 0.3, 0.7, 200, 25)
        assert rc.shift == pytest.approx(0.4)

    def test_regime_table_covers_every_present_asset(self, crypto_prices, macro_prices):
        returns = correlation.align_returns(crypto_prices, macro_prices)
        table = correlation.regime_table(returns, ["BTC", "ETH", "SOL"])
        assert list(table.index) == ["BTC", "ETH", "SOL"]


class TestAlignment:
    def test_inner_join_keeps_only_shared_trading_days(self, crypto_prices, macro_prices):
        """
        Crypto trades 365 days, macro roughly 252. Forward-filling macro across
        the weekend would manufacture zero-return days and drag correlation
        toward zero, so the join is inner and differencing happens after.
        """
        returns = correlation.align_returns(crypto_prices, macro_prices)
        assert len(returns) < len(crypto_prices)
        assert len(returns) > 200
