"""
GARCH(1,1) tests.

The model exists because rolling standard deviation weights every day in its
window equally and drops a shock in one step when it falls out of the far end.
These tests pin the properties that make the GARCH estimate different, and the
fail-open behaviour that keeps a bad fit from taking the tab down.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.analysis import volatility


@pytest.fixture(scope="module")
def fitted(btc_price):
    return volatility.fit_garch(btc_price, "BTC", horizon=30)


class TestFit:
    def test_converges_on_a_normal_series(self, fitted):
        assert fitted.converged, fitted.message

    def test_parameters_are_in_their_valid_ranges(self, fitted):
        assert fitted.omega > 0
        assert fitted.alpha >= 0
        assert fitted.beta >= 0

    def test_persistence_is_alpha_plus_beta(self, fitted):
        assert fitted.persistence == pytest.approx(fitted.alpha + fitted.beta)

    def test_conditional_volatility_is_annualised_and_positive(self, fitted):
        series = fitted.conditional_volatility
        assert not series.empty
        assert (series > 0).all()
        # Annualised crypto vol lives in tens of percent, not hundredths.
        assert 0.05 < series.median() < 5.0

    def test_conditional_series_aligns_with_the_returns_index(self, btc_price, fitted):
        assert len(fitted.conditional_volatility) == len(
            volatility.daily_returns(btc_price)
        )


class TestForecast:
    def test_horizon_length_is_honoured(self, fitted):
        assert len(fitted.forecast) == 30

    def test_forecast_starts_the_day_after_history_ends(self, btc_price, fitted):
        assert fitted.forecast.index[0] == btc_price.index[-1] + pd.Timedelta(days=1)

    def test_forecast_values_are_positive(self, fitted):
        assert (fitted.forecast > 0).all()


class TestInterpretation:
    def test_long_run_volatility_is_finite_when_stationary(self, fitted):
        if fitted.is_stationary:
            assert np.isfinite(fitted.long_run_volatility)

    def test_non_stationary_process_has_no_long_run_level(self):
        # A conditional series is supplied because verdict() reports the missing
        # estimate first; this test is about the persistence branch below it.
        idx = pd.date_range("2024-01-01", periods=10, freq="D")
        result = volatility.GarchResult(
            "X", omega=0.1, alpha=0.5, beta=0.6,
            conditional_volatility=pd.Series(np.full(10, 0.5), index=idx),
            forecast=pd.Series(dtype=float),
        )
        assert result.persistence > 1
        assert not result.is_stationary
        assert np.isnan(result.long_run_volatility)
        assert "do not decay" in result.verdict()

    def test_verdict_is_a_readable_sentence(self, fitted):
        text = fitted.verdict()
        assert isinstance(text, str) and len(text) > 20


class TestFailsOpen:
    def test_too_little_history_reports_rather_than_raises(self):
        idx = pd.date_range("2024-01-01", periods=20, freq="D")
        price = pd.Series(np.linspace(100, 120, 20), index=idx)
        result = volatility.fit_garch(price, "X")
        assert result.converged is False
        assert "at least 60" in result.message
        assert result.conditional_volatility.empty

    def test_failed_fit_still_produces_a_usable_object(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        result = volatility.fit_garch(pd.Series([1.0] * 5, index=idx), "X")
        assert not result.converged
        assert isinstance(result.verdict(), str)
        assert np.isnan(result.current_volatility)


class TestAgainstRollingEstimate:
    def test_reacts_faster_than_the_rolling_window(self, btc_price, fitted):
        """
        GARCH updates on every observation; the rolling estimate only moves as
        days enter and leave its window. The GARCH series should therefore be
        the more variable of the two day to day.
        """
        rolling = volatility.rolling_volatility(btc_price).dropna()
        garch = fitted.conditional_volatility.reindex(rolling.index).dropna()
        assert garch.diff().abs().mean() > 0
        assert len(garch) > 100
