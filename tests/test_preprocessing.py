"""
Preprocessing tests.

Each one pins a specific way a provider has actually misbehaved: repeated
timestamps, values delivered as strings, intraday points that need collapsing,
and columns that arrive entirely empty after a rate limit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import preprocessing as pp


def _frame(values, start="2024-01-01", freq="D") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(values), freq=freq)
    return pd.DataFrame({"price": values}, index=idx)


class TestIndexHygiene:
    def test_duplicate_timestamps_are_removed_and_counted(self):
        """A repeated day double-counts in every statistic downstream."""
        idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"])
        frame = pd.DataFrame({"price": [1.0, 2.0, 2.5, 3.0]}, index=idx)
        out, report = pp.clean_timeseries(frame, "test", resample_daily=False)
        assert report.duplicates_removed == 1
        assert not out.index.duplicated().any()

    def test_last_duplicate_wins(self):
        """Providers resend a day to correct it, so the later value is the fix."""
        idx = pd.to_datetime(["2024-01-01", "2024-01-01"])
        frame = pd.DataFrame({"price": [1.0, 9.0]}, index=idx)
        out, _ = pp.clean_timeseries(frame, "test", resample_daily=False)
        assert out["price"].iloc[0] == 9.0

    def test_unsorted_index_is_sorted(self):
        idx = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"])
        frame = pd.DataFrame({"price": [3.0, 1.0, 2.0]}, index=idx)
        out, _ = pp.clean_timeseries(frame, "test", resample_daily=False)
        assert out.index.is_monotonic_increasing

    def test_unparseable_timestamps_are_dropped(self):
        frame = pd.DataFrame({"price": [1.0, 2.0]}, index=["2024-01-01", "not a date"])
        out, _ = pp.clean_timeseries(frame, "test", resample_daily=False)
        assert len(out) == 1


class TestTypeCoercion:
    def test_string_values_are_coerced_and_counted(self):
        """
        alternative.me returns the Fear & Greed value as a string. Comparing
        strings numerically does not raise — it silently produces nonsense.
        """
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        frame = pd.DataFrame({"fng_value": ["30", "45", "60"]}, index=idx)
        out, report = pp.clean_timeseries(frame, "sentiment")
        assert pd.api.types.is_numeric_dtype(out["fng_value"])
        assert report.non_numeric_coerced == 3

    def test_sentiment_values_are_clipped_to_the_documented_range(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        frame = pd.DataFrame({"fng_value": [-5.0, 50.0, 130.0]}, index=idx)
        out, report = pp.clean_sentiment_frame(frame)
        assert out["fng_value"].between(0, 100).all()
        assert report.outliers_flagged >= 2


class TestResamplingAndGaps:
    def test_intraday_points_collapse_to_one_row_per_day(self):
        """CoinGecko returns intraday points on short windows."""
        idx = pd.to_datetime(
            ["2024-01-01 00:00", "2024-01-01 12:00", "2024-01-02 06:00"]
        )
        frame = pd.DataFrame({"price": [1.0, 2.0, 3.0]}, index=idx)
        out, _ = pp.clean_timeseries(frame, "crypto")
        assert len(out) == 2
        assert out["price"].iloc[0] == 2.0  # last value of the day

    def test_gaps_are_forward_filled_and_counted(self):
        frame = _frame([1.0, np.nan, 3.0])
        out, report = pp.clean_timeseries(frame, "test")
        assert report.missing_filled == 1
        assert out["price"].iloc[1] == 1.0

    def test_fill_is_forward_only_never_backward(self):
        """
        Back-filling would carry a later observation into an earlier day, which
        is look-ahead: the model would see information from the future.
        """
        frame = _frame([np.nan, 2.0, 3.0])
        out, _ = pp.clean_timeseries(frame, "test")
        assert 2.0 not in out["price"].to_numpy()[:1] or len(out) == 2

    def test_leading_all_nan_rows_are_dropped(self):
        frame = _frame([np.nan, 2.0, 3.0])
        out, _ = pp.clean_timeseries(frame, "test")
        assert out["price"].notna().all()


class TestImpossibleValues:
    def test_non_positive_prices_are_removed_when_required(self):
        """A zero price becomes an infinity the moment a return is taken."""
        frame = _frame([100.0, 0.0, 102.0])
        out, report = pp.clean_timeseries(frame, "crypto", require_positive=True)
        assert report.non_positive_removed == 1
        assert (out["price"] > 0).all()

    def test_negative_values_allowed_where_legitimate(self):
        """A yield can be zero or negative; a price cannot."""
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        frame = pd.DataFrame({"US 10Y Yield": [0.5, -0.1, 0.2]}, index=idx)
        out, report = pp.clean_macro_frame(frame)
        assert report.non_positive_removed == 0
        assert len(out) == 3


class TestOutliers:
    def test_outliers_are_flagged_not_deleted(self):
        """
        VaR and CVaR exist to describe the tail. Trimming it before measuring
        would report a comfortable number that is wrong in the one direction
        that matters.
        """
        rng = np.random.default_rng(0)
        values = 100 * np.cumprod(1 + rng.normal(0, 0.01, 200))
        values[150] *= 3.0  # a spike no real market produced
        out, report = pp.clean_timeseries(_frame(values), "test")
        assert report.outliers_flagged >= 1
        assert len(out) == 200, "an outlier was removed rather than flagged"

    def test_clean_series_flags_nothing(self):
        rng = np.random.default_rng(1)
        values = 100 * np.cumprod(1 + rng.normal(0, 0.01, 200))
        _, report = pp.clean_timeseries(_frame(values), "test")
        assert report.outliers_flagged == 0

    def test_short_series_is_not_judged(self):
        assert pp.count_outliers(_frame([1.0, 2.0, 3.0])) == 0


class TestReporting:
    def test_empty_input_reports_zero_rows_without_raising(self):
        out, report = pp.clean_timeseries(pd.DataFrame(), "test")
        assert out.empty
        assert report.rows_out == 0
        assert not report.is_healthy

    def test_coverage_measured_against_the_requested_window(self):
        _, report = pp.clean_timeseries(_frame([1.0] * 180), "test", expected_days=365)
        assert report.coverage == pytest.approx(180 / 365)

    def test_business_day_feed_is_not_penalised_for_weekends(self):
        """~70% coverage is correct for a market that shuts at weekends."""
        idx = pd.bdate_range("2024-01-01", periods=252)
        frame = pd.DataFrame({"px": np.arange(252.0)}, index=idx)
        _, report = pp.clean_macro_frame(frame, expected_days=365)
        assert report.is_healthy

    def test_summary_is_human_readable(self):
        _, report = pp.clean_timeseries(_frame([1.0] * 30), "Crypto — BTC")
        text = report.summary()
        assert "Crypto — BTC" in text and "30 rows" in text

    def test_quality_table_has_one_row_per_report(self):
        reports = [
            pp.clean_timeseries(_frame([1.0] * 10), f"feed {i}")[1] for i in range(3)
        ]
        table = pp.quality_table(reports)
        assert len(table) == 3
        assert "Status" in table.columns

    def test_quality_table_of_nothing_is_empty_not_an_error(self):
        assert pp.quality_table([]).empty
