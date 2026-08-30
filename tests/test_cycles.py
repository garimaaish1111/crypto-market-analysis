"""
Regression tests for the market-cycle module.

Each test here pins a defect described in CHANGES.md. Before these existed the
evidence for those fixes was prose in a changelog; now it is executable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.analysis import cycles


# --------------------------------------------------------------------------- #
# RSI -- CHANGES.md section 7, "RSI returned 50 for maximum strength"
# --------------------------------------------------------------------------- #
class TestRSI:
    def test_unbroken_gains_return_100_not_50(self, rising):
        """
        The original bug: an average loss of zero made RS NaN, and .fillna(50)
        then reported the strongest possible rally as neutral momentum.
        """
        value = cycles.rsi(rising).dropna().iloc[-1]
        assert value == pytest.approx(100.0), "a series that only rises is maximum strength"

    def test_flat_series_is_neutral_50(self, flat):
        """Zero gain and zero loss is the one case that genuinely is neutral."""
        value = cycles.rsi(flat).dropna().iloc[-1]
        assert value == pytest.approx(50.0)

    def test_unbroken_losses_return_0(self, falling):
        value = cycles.rsi(falling).dropna().iloc[-1]
        assert value == pytest.approx(0.0, abs=1e-9)

    def test_warmup_window_is_nan_not_fabricated(self, btc_price):
        """
        Days inside the warm-up have no RSI. Filling them with 50 would invent
        momentum readings for days that have none.
        """
        out = cycles.rsi(btc_price, window=config.RSI_WINDOW)
        assert out.iloc[: config.RSI_WINDOW].isna().all()
        assert out.iloc[config.RSI_WINDOW:].notna().any()

    def test_uses_wilder_smoothing_not_pandas_default(self, btc_price):
        """
        ewm defaults to adjust=True, which is a different weighting scheme.
        Recomputing with adjust=True must give a different answer, proving the
        implementation is not silently sitting on the pandas default.

        The two schemes converge as the series lengthens -- by day 365 they agree
        to eleven decimal places -- so the comparison has to be made just after
        the warm-up, which is exactly where the weighting difference bites.
        """
        delta = btc_price.diff()
        gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
        w = config.RSI_WINDOW
        wrong_g = gain.ewm(alpha=1 / w, min_periods=w, adjust=True).mean()
        wrong_l = loss.ewm(alpha=1 / w, min_periods=w, adjust=True).mean()
        wrong = 100 - 100 / (1 + wrong_g / wrong_l.replace(0, np.nan))

        actual = cycles.rsi(btc_price)
        first = w + 1  # first index with a value under both schemes
        assert not np.isclose(actual.iloc[first], wrong.iloc[first], atol=1e-6)

    def test_bounded_between_0_and_100(self, btc_price):
        out = cycles.rsi(btc_price).dropna()
        assert out.between(0, 100).all()


# --------------------------------------------------------------------------- #
# Adaptive moving averages -- CHANGES.md section 1
# --------------------------------------------------------------------------- #
class TestMovingAverageWindows:
    @pytest.mark.parametrize(
        "n_obs,expected",
        [(90, (7, 30)), (180, (15, 60)), (270, (22, 90)), (365, (50, 200))],
    )
    def test_windows_match_documented_table(self, n_obs, expected):
        """The exact pairs quoted in the CHANGES.md table and in the dashboard."""
        assert config.ma_windows(n_obs) == expected

    def test_canonical_pair_kept_when_history_allows(self):
        assert config.ma_windows(400) == (config.MA_SHORT, config.MA_LONG)

    def test_ratio_preserved_when_scaled_down(self):
        short, long = config.ma_windows(120)
        assert long == pytest.approx(short * 4, rel=0.35)

    def test_short_window_produces_labels(self):
        """
        The defect: on a 90-day window a 200-day MA yields nothing, so every day
        came back Undetermined and the metric rendered as nan%.
        """
        from src.data import sample_data

        price = sample_data.crypto_ohlcv("BTC", 90)["price"]
        phases = cycles.phase_series(price)
        labelled = (phases != "Undetermined").mean()
        assert labelled > 0.5, f"only {labelled:.0%} of a 90-day window is labelled"


# --------------------------------------------------------------------------- #
# Phase rules -- CHANGES.md section 8, "Phase rules tightened"
# --------------------------------------------------------------------------- #
class TestPhaseRules:
    @staticmethod
    def _series(values) -> pd.Series:
        idx = pd.date_range("2024-01-01", periods=len(values), freq="D")
        return pd.Series(values, index=idx, dtype=float)

    def test_distribution_requires_price_near_the_peak(self):
        """
        Distribution describes supply meeting demand near a top. An uptrend with
        soft momentum far below the peak is a Transition, not Distribution.
        """
        values = (
            list(np.linspace(100, 300, 120))
            + list(np.linspace(300, 180, 60))
            + list(np.linspace(180, 200, 120))
        )
        price = self._series(values)
        phases = cycles.phase_series(price)
        dd = price / price.cummax() - 1
        far_off_highs = phases[(dd < cycles.NEAR_HIGHS) & (phases != "Undetermined")]
        assert "Distribution" not in set(far_off_highs), (
            "Distribution fired while price was more than 10% below the peak"
        )

    def test_shallow_downtrend_is_markdown_not_accumulation(self):
        """
        Accumulation is a basing bottom. An early, shallow decline is not one --
        the old rule labelled any downtrend Accumulation.
        """
        values = list(np.linspace(100, 300, 150)) + list(np.linspace(300, 280, 150))
        price = self._series(values)
        phases = cycles.phase_series(price)
        dd = price / price.cummax() - 1
        shallow = phases[(dd > cycles.DEEP_DRAWDOWN) & (phases != "Undetermined")]
        assert "Accumulation" not in set(shallow), (
            "Accumulation fired on a drawdown shallower than 25%"
        )

    def test_strong_uptrend_is_markup(self, rising):
        phases = cycles.phase_series(rising, short=5, long=20)
        assert phases.iloc[-1] == "Markup (Bull)"

    def test_every_label_is_a_known_phase(self, btc_price):
        known = set(config.PHASE_COLORS) | {"Undetermined"}
        assert set(cycles.phase_series(btc_price).unique()) <= known

    def test_phase_series_is_index_aligned(self, btc_price):
        phases = cycles.phase_series(btc_price)
        assert phases.index.equals(btc_price.index)
        assert len(phases) == len(btc_price)

    def test_undetermined_exactly_where_inputs_are_missing(self, btc_price):
        """Labels are withheld precisely when the long MA or RSI has no value."""
        short, long = config.ma_windows(len(btc_price))
        ma = cycles.moving_averages(btc_price, short, long)
        momentum = cycles.rsi(btc_price)
        expected = ma["ma_long"].isna() | momentum.isna()
        actual = cycles.phase_series(btc_price) == "Undetermined"
        assert (expected.to_numpy() == actual.to_numpy()).all()


# --------------------------------------------------------------------------- #
# current_phase summary object
# --------------------------------------------------------------------------- #
class TestCurrentPhase:
    def test_reports_the_windows_actually_used(self, btc_price):
        """
        The label must never be computed on a different basis than the reader is
        shown, so the windows in use travel with the result.
        """
        phase = cycles.current_phase("BTC", btc_price)
        assert (phase.ma_short, phase.ma_long) == config.ma_windows(len(btc_price))

    def test_drawdown_is_negative_or_zero(self, btc_price):
        assert cycles.current_phase("BTC", btc_price).drawdown_from_high <= 0

    def test_as_dict_labels_drawdown_as_window_high(self, btc_price):
        """
        CHANGES.md section 7: cummax() over a 365-day window is a one-year high,
        not an all-time high, and the label has to say so.
        """
        keys = cycles.current_phase("BTC", btc_price).as_dict()
        assert "Drawdown vs window high" in keys
        assert not any("ATH" in k for k in keys)
