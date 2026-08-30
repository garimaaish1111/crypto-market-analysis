"""
Currency conversion and formatting.

Indian digit grouping is the part most likely to be got wrong and the part an
Indian examiner spots instantly: the last three digits group together, then
everything above that pairs off. 7456772 is 74,56,772 — not 7,456,772.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.data import fx

INR = config.CURRENCY == "inr"


class TestIndianDigitGrouping:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (7456772, "₹74,56,772"),
            (100000, "₹1,00,000"),
            (1000000, "₹10,00,000"),
            (10000000, "₹1,00,00,000"),
            (999, "₹999.00"),
            (1000, "₹1,000"),
            (12345678901, "₹12,34,56,78,901"),
        ],
    )
    @pytest.mark.skipif(not INR, reason="only meaningful in rupee mode")
    def test_grouping_matches_indian_convention(self, value, expected):
        assert fx.format_currency(value) == expected

    @pytest.mark.skipif(not INR, reason="only meaningful in rupee mode")
    def test_western_grouping_is_not_used(self):
        """The failure this guards: ₹7,456,772 instead of ₹74,56,772."""
        assert fx.format_currency(7456772) != "₹7,456,772"

    @pytest.mark.skipif(not INR, reason="only meaningful in rupee mode")
    def test_negative_values_keep_the_sign_outside_the_symbol(self):
        assert fx.format_currency(-100000) == "-₹1,00,000"

    def test_nan_renders_as_a_dash_not_a_crash(self):
        assert fx.format_currency(float("nan")) == "—"
        assert fx.compact_currency(float("nan")) == "—"

    def test_small_values_keep_decimals(self):
        """XRP and ADA genuinely trade below ₹100; rounding them to 0 is wrong."""
        assert "0.85" in fx.format_currency(0.85)
        assert "0.85" in fx.compact_currency(0.85)


class TestCompactForm:
    @pytest.mark.skipif(not INR, reason="only meaningful in rupee mode")
    def test_uses_lakh_and_crore(self):
        assert fx.compact_currency(7456772).endswith(" L")
        assert fx.compact_currency(123456789).endswith(" Cr")

    @pytest.mark.skipif(not INR, reason="only meaningful in rupee mode")
    def test_crore_threshold_is_ten_million(self):
        assert " L" in fx.compact_currency(9_999_999)
        assert " Cr" in fx.compact_currency(10_000_001)


class TestMacroConversion:
    @staticmethod
    def _frame() -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        return pd.DataFrame(
            {
                "S&P 500": np.full(5, 6000.0),
                "Gold": np.full(5, 3000.0),
                "Crude Oil": np.full(5, 80.0),
                "US Dollar Index": np.full(5, 103.0),
                "US 10Y Yield": np.full(5, 4.1),
            },
            index=idx,
        )

    @pytest.fixture
    def rate(self) -> pd.Series:
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        return pd.Series(np.full(5, 90.0), index=idx, name="usd_inr")

    @pytest.mark.skipif(not INR, reason="only meaningful in rupee mode")
    def test_dollar_priced_columns_are_converted(self, rate):
        out = fx.convert_macro(self._frame(), rate)
        assert out["S&P 500"].iloc[0] == pytest.approx(6000 * 90)
        assert out["Gold"].iloc[0] == pytest.approx(3000 * 90)
        assert out["Crude Oil"].iloc[0] == pytest.approx(80 * 90)

    @pytest.mark.skipif(not INR, reason="only meaningful in rupee mode")
    def test_index_and_rate_columns_are_left_alone(self, rate):
        """
        The US Dollar Index is an index level and the 10Y yield is a rate in
        percentage points. Neither is a price in dollars, so multiplying either
        by an exchange rate would be meaningless.
        """
        out = fx.convert_macro(self._frame(), rate)
        assert out["US Dollar Index"].iloc[0] == pytest.approx(103.0)
        assert out["US 10Y Yield"].iloc[0] == pytest.approx(4.1)

    def test_conversion_preserves_returns_within_a_column(self, rate):
        """A constant multiplier changes no percentage return."""
        frame = self._frame()
        frame["Gold"] = [3000, 3060, 3000, 2940, 3000]
        out = fx.convert_macro(frame, rate)
        assert np.allclose(
            frame["Gold"].pct_change().dropna(),
            out["Gold"].pct_change().dropna(),
        )

    def test_empty_frame_is_returned_untouched(self, rate):
        assert fx.convert_macro(pd.DataFrame(), rate).empty

    def test_missing_rate_falls_back_rather_than_producing_nan(self):
        """A dead FX feed must cost accuracy, not the whole macro frame."""
        empty_rate = pd.Series(dtype=float)
        out = fx.convert_macro(self._frame(), empty_rate)
        assert out["S&P 500"].notna().all()


class TestFxLoader:
    def test_latest_rate_falls_back_when_series_is_empty(self):
        expected = config.USD_INR_FALLBACK if INR else 1.0
        assert fx.latest_rate(pd.Series(dtype=float)) == pytest.approx(expected)
        assert fx.latest_rate(None) == pytest.approx(expected)

    def test_latest_rate_reads_the_last_observation(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        assert fx.latest_rate(pd.Series([90.0, 92.0, 95.0], index=idx)) == pytest.approx(95.0)


class TestNoHardcodedCurrencySymbols:
    def test_app_and_charts_use_the_config_symbol(self):
        """
        Guards the whole point of the currency layer: a stray "$" in a format
        string silently mislabels every rupee figure on the page.
        """
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        for relative in ("app.py", "src/viz/charts.py"):
            source = (root / relative).read_text(encoding="utf8")
            code = "\n".join(
                line for line in source.splitlines()
                if not line.strip().startswith("#")
            )
            assert 'f"${' not in code, f"hardcoded dollar format in {relative}"
            assert 'f"₹{' not in code, f"hardcoded rupee format in {relative}"
