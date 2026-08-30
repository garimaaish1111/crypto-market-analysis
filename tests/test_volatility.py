"""
Risk-metric tests.

The Sortino and downside-deviation cases pin CHANGES.md section 7, where the
original used the standard deviation of the losing days -- a different quantity
from downside deviation, and one that inflates the ratio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.analysis import volatility


class TestDownsideDeviation:
    def test_matches_hand_computed_value(self):
        """
        sqrt(mean(min(r - target, 0)^2)) over ALL observations.
        For [-0.02, 0.05, -0.04, 0.03] against target 0:
        shortfalls are [-0.02, 0, -0.04, 0]; mean square = 0.0005; sqrt = 0.02236
        """
        rets = pd.Series([-0.02, 0.05, -0.04, 0.03])
        assert volatility.downside_deviation(rets, target=0.0) == pytest.approx(
            np.sqrt((0.0004 + 0.0016) / 4)
        )

    def test_differs_from_std_of_losses_only(self):
        """
        The old shortcut. It measures spread about the mean of the losses rather
        than shortfall against the target, and is a smaller number here -- which
        is precisely why it inflated the Sortino ratio.
        """
        rets = pd.Series([-0.02, 0.05, -0.04, 0.03])
        correct = volatility.downside_deviation(rets, target=0.0)
        old_shortcut = float(rets[rets < 0].std())
        assert not np.isclose(correct, old_shortcut)
        assert correct > old_shortcut

    def test_all_gains_gives_zero_downside(self):
        rets = pd.Series([0.01, 0.02, 0.03])
        assert volatility.downside_deviation(rets, target=0.0) == pytest.approx(0.0)

    def test_counts_all_observations_not_just_losers(self):
        """
        Adding winning days must lower downside deviation, because the mean is
        taken across every observation. Under the old shortcut they would be
        discarded and the answer would not move.
        """
        losers = pd.Series([-0.02, -0.04])
        padded = pd.Series([-0.02, -0.04, 0.05, 0.05, 0.05])
        assert volatility.downside_deviation(padded, 0.0) < volatility.downside_deviation(
            losers, 0.0
        )


class TestSortino:
    def test_lower_than_naive_version_using_loss_std(self, btc_price):
        """
        The corrected denominator is larger, so the corrected Sortino must be
        smaller than the inflated original. This is the regression that matters.
        """
        rets = volatility.daily_returns(btc_price)
        target = config.RISK_FREE_RATE / config.TRADING_DAYS
        inflated = float(
            (rets - target).mean() / rets[rets < 0].std() * np.sqrt(config.TRADING_DAYS)
        )
        assert volatility.sortino_ratio(btc_price) < inflated

    def test_exceeds_sharpe_for_a_positive_drift_series(self, btc_price):
        """
        Sortino ignores upside volatility, so for an asset that rose over the
        window it should read higher than Sharpe.
        """
        assert volatility.sortino_ratio(btc_price) > volatility.sharpe_ratio(btc_price)


class TestVaR:
    def test_var_99_is_worse_than_var_95(self, btc_price):
        assert volatility.historical_var(btc_price, 0.99) > volatility.historical_var(
            btc_price, 0.95
        )

    def test_cvar_is_worse_than_var(self, btc_price):
        """Expected shortfall averages the tail beyond the threshold, so it is deeper."""
        assert volatility.conditional_var(btc_price, 0.95) > volatility.historical_var(
            btc_price, 0.95
        )

    def test_var_returned_as_positive_loss(self, btc_price):
        assert volatility.historical_var(btc_price, 0.95) > 0

    def test_var_matches_empirical_percentile(self):
        """Non-parametric by construction -- no normal-distribution assumption."""
        idx = pd.date_range("2024-01-01", periods=101, freq="D")
        rets = np.linspace(-0.10, 0.10, 100)
        price = pd.Series(np.concatenate([[100.0], 100 * np.cumprod(1 + rets)]), index=idx)
        observed = volatility.daily_returns(price)
        assert volatility.historical_var(price, 0.95) == pytest.approx(
            -np.percentile(observed, 5)
        )

    def test_empty_series_returns_nan_not_crash(self):
        empty = pd.Series([1.0])
        assert np.isnan(volatility.historical_var(empty))
        assert np.isnan(volatility.conditional_var(empty))


class TestDrawdown:
    def test_monotonic_rise_has_no_drawdown(self, rising):
        assert volatility.max_drawdown(rising) == pytest.approx(0.0)

    def test_known_drawdown_is_exact(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="D")
        price = pd.Series([100.0, 200.0, 100.0, 150.0], index=idx)
        assert volatility.max_drawdown(price) == pytest.approx(-0.5)

    def test_drawdown_series_never_positive(self, btc_price):
        assert (volatility.drawdown_series(btc_price) <= 1e-12).all()


class TestAnnualisation:
    def test_uses_365_not_252(self, btc_price):
        """
        Crypto trades every day of the year, so annualising on 252 understates
        volatility by about 20%.
        """
        assert config.TRADING_DAYS == 365
        daily_std = volatility.daily_returns(btc_price).std()
        assert volatility.annualised_volatility(btc_price) == pytest.approx(
            daily_std * np.sqrt(365)
        )

    def test_rolling_vol_window_length(self, btc_price):
        rv = volatility.rolling_volatility(btc_price, window=30)
        assert rv.iloc[:29].isna().all()
        assert rv.dropna().gt(0).all()


class TestVolatilityRegime:
    def test_returns_a_known_label(self, btc_price):
        assert volatility.volatility_regime(btc_price) in {"Low", "Elevated", "High"}

    def test_unknown_when_history_too_short(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="D")
        assert volatility.volatility_regime(pd.Series(np.arange(10.0) + 100, index=idx)) == "Unknown"


class TestRiskProfile:
    def test_all_fields_populated(self, btc_price):
        profile = volatility.risk_profile("BTC", btc_price)
        assert profile.symbol == "BTC"
        for field in ("annual_volatility", "var_95", "var_99", "cvar_95", "max_drawdown"):
            assert np.isfinite(getattr(profile, field))

    def test_as_dict_renders_nan_as_dash_not_crash(self):
        profile = volatility.RiskProfile(
            "X", float("nan"), float("nan"), float("nan"), float("nan"),
            float("nan"), float("nan"), float("nan"), "Unknown",
        )
        assert profile.as_dict()["Annual Vol"] == "—"
