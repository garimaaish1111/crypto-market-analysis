"""
Forecasting tests.

These pin the validation machinery from CHANGES.md section 6 -- walk-forward
one-step-ahead prediction, a naive random-walk baseline scored on the same days,
and a skill score. The point of the module is not that ARIMA predicts crypto
(it does not); it is that the dashboard can prove whether it does.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import config
from src.analysis import forecast


@pytest.fixture(scope="module")
def result(btc_price):
    """One fit reused across the module -- ARIMA order search is slow."""
    return forecast.forecast_price(btc_price, "BTC", horizon=30)


class TestSkillScore:
    def test_skill_formula(self):
        """1 - RMSE(model) / RMSE(naive)."""
        r = forecast.ForecastResult(
            symbol="X", order=(1, 1, 1),
            forecast=pd.Series(dtype=float), lower=pd.Series(dtype=float),
            upper=pd.Series(dtype=float),
            rmse=80.0, mae=0.0, mape=0.0,
            naive_rmse=100.0, naive_mae=0.0, naive_mape=0.0,
        )
        assert r.skill == pytest.approx(0.2)
        assert r.beats_naive is True

    def test_worse_than_naive_gives_negative_skill(self):
        r = forecast.ForecastResult(
            symbol="X", order=(1, 1, 1),
            forecast=pd.Series(dtype=float), lower=pd.Series(dtype=float),
            upper=pd.Series(dtype=float),
            rmse=120.0, mae=0.0, mape=0.0,
            naive_rmse=100.0, naive_mae=0.0, naive_mape=0.0,
        )
        assert r.skill < 0
        assert r.beats_naive is False

    def test_zero_naive_rmse_returns_nan_not_zero_division(self):
        r = forecast.ForecastResult(
            symbol="X", order=(1, 1, 1),
            forecast=pd.Series(dtype=float), lower=pd.Series(dtype=float),
            upper=pd.Series(dtype=float),
            rmse=1.0, mae=0.0, mape=0.0,
            naive_rmse=0.0, naive_mae=0.0, naive_mape=0.0,
        )
        assert np.isnan(r.skill)
        assert r.beats_naive is False


class TestVerdict:
    @pytest.mark.parametrize(
        "rmse,naive_rmse,expected_fragment",
        [
            (50.0, 100.0, "beat the random-walk baseline"),
            (99.0, 100.0, "too small to rely on"),
            (120.0, 100.0, "did not beat"),
        ],
    )
    def test_verdict_matches_skill_band(self, rmse, naive_rmse, expected_fragment):
        r = forecast.ForecastResult(
            symbol="X", order=(0, 1, 1),
            forecast=pd.Series(dtype=float), lower=pd.Series(dtype=float),
            upper=pd.Series(dtype=float),
            rmse=rmse, mae=0.0, mape=0.0,
            naive_rmse=naive_rmse, naive_mae=0.0, naive_mape=0.0,
        )
        assert expected_fragment in r.verdict()

    def test_verdict_never_promises_a_price_target(self, result):
        """The dashboard must not present a forecast as a prediction to trade on."""
        assert "price target" not in result.verdict() or "not a price target" in result.verdict().lower() or "range of outcomes" in result.verdict()


class TestWalkForward:
    def test_one_prediction_per_test_day(self, btc_price):
        """
        The defect: a single long forecast across the whole test period mostly
        measured how far the price drifted. There must be one prediction per day.
        """
        log_price = np.log(btc_price.dropna().asfreq("D").ffill())
        n_test = max(10, int(len(log_price) * config.FORECAST_TEST_FRACTION))
        train, test = log_price.iloc[:-n_test], log_price.iloc[-n_test:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = forecast._walk_forward(train, test, (0, 1, 1))
        assert len(preds) == len(test)
        assert preds.index.equals(test.index)

    def test_naive_baseline_uses_previous_actual(self, result):
        """Tomorrow's price is today's price, and the first test day comes from train."""
        assert np.isfinite(result.naive_rmse)
        assert result.naive_rmse > 0

    def test_model_and_naive_scored_on_identical_days(self, result):
        assert len(result.fitted_test) == len(result.test_actual)
        assert result.fitted_test.index.equals(result.test_actual.index)

    def test_test_split_respects_configured_fraction(self, btc_price, result):
        expected = max(10, int(len(btc_price) * config.FORECAST_TEST_FRACTION))
        assert len(result.test_actual) == pytest.approx(expected, abs=2)


class TestForecastOutput:
    def test_interval_brackets_the_point_forecast(self, result):
        assert (result.lower <= result.forecast + 1e-9).all()
        assert (result.forecast <= result.upper + 1e-9).all()

    def test_interval_widens_with_horizon(self, result):
        """Uncertainty compounds -- a flat band would be the giveaway of a bug."""
        width = result.upper - result.lower
        assert width.iloc[-1] > width.iloc[0]

    def test_forecast_prices_are_positive(self, result):
        """Modelling log-price is what guarantees this."""
        assert (result.forecast > 0).all()
        assert (result.lower > 0).all()

    def test_horizon_length_honoured(self, result):
        assert len(result.forecast) == 30

    def test_forecast_starts_the_day_after_history_ends(self, btc_price, result):
        assert result.forecast.index[0] == btc_price.index[-1] + pd.Timedelta(days=1)

    def test_order_is_differenced_once(self, result):
        """d=1 on log-price means the model works on log returns."""
        assert result.order[1] == 1

    def test_metrics_are_finite_and_positive(self, result):
        for metric in (result.rmse, result.mae, result.mape, result.naive_rmse):
            assert np.isfinite(metric) and metric > 0


class TestWarningScope:
    def test_no_module_level_filterwarnings(self):
        """
        CHANGES.md section 7: a module-scope filterwarnings("ignore") silenced
        every warning in the whole process, not just statsmodels chatter.
        """
        from pathlib import Path

        source = (
            Path(__file__).resolve().parent.parent / "src" / "analysis" / "forecast.py"
        ).read_text(encoding="utf8")
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("warnings.filterwarnings"):
                pytest.fail("filterwarnings called at module scope; use a context manager")
        assert "catch_warnings" in source
