"""
Chart tests.

These are not pixel comparisons. They assert the two properties that actually
broke in this project: every figure builds without raising, and no colour
outside the declared palette reaches the screen.

CHANGES.md section 11 records that Plotly's stock templates embed their own
colourways and per-trace colour scales -- the Plasma ramp for heatmaps, a
ten-hue qualitative set for lines -- so any trace that did not set a colour
explicitly pulled from those. The custom template exists to prevent that, and
the palette test is what keeps it honest as new charts are added.
"""
from __future__ import annotations

import json
import re

import pandas as pd
import pytest

import config
from src.analysis import correlation, cycles, forecast
from src.viz import charts

HEX = re.compile(r"#[0-9a-fA-F]{6}")


def _hex_values(fig) -> set[str]:
    """Every hex colour appearing anywhere in a figure's JSON."""
    return {m.upper() for m in HEX.findall(json.dumps(fig.to_plotly_json(), default=str))}


@pytest.fixture(scope="module")
def allowed() -> set[str]:
    """The five palette colours, the two tints, and the theme tokens they derive."""
    values = {
        config.INK, config.ACCENT, config.ACCENT_DARK, config.MUTED, config.MUTED_LIGHT,
        config.PAGE_BG, config.PANEL_BG, config.TEXT, config.TEXT_MUTED, config.BORDER,
        config.SERIES_PRIMARY, config.SERIES_TERTIARY, config.SURFACE, config.CELL_TEXT,
        config.UNDETERMINED_COLOR, "#ECEFF1", "#CFD8DC", "#FFFFFF",
    }
    values |= set(config.PHASE_COLORS.values())
    values |= {c for _, c in config.CORRELATION_SCALE if isinstance(c, str)}
    return {v.upper() for v in values if isinstance(v, str) and v.startswith("#")}


@pytest.fixture(scope="module")
def figures(btc_price, crypto_prices, macro_prices):
    """One of every chart the dashboard can render."""
    from src.data import sample_data

    df = sample_data.crypto_ohlcv("BTC", 365)
    returns = correlation.align_returns(crypto_prices, macro_prices)
    result = forecast.forecast_price(btc_price, "BTC", horizon=30)

    return {
        "price_with_ma": charts.price_with_ma(btc_price, "BTC"),
        "volume_bars": charts.volume_bars(df, "BTC"),
        "rolling_vol": charts.rolling_vol_chart(btc_price, "BTC"),
        "drawdown": charts.drawdown_chart(btc_price, "BTC"),
        "phase_ribbon": charts.phase_ribbon(btc_price, "BTC"),
        "phase_distribution": charts.phase_distribution(cycles.phase_series(btc_price)),
        "correlation_heatmap": charts.correlation_heatmap(
            correlation.correlation_matrix(returns)
        ),
        "rolling_corr": charts.rolling_corr_chart(
            correlation.rolling_correlation(returns, "BTC", "S&P 500", 30), "BTC", "S&P 500"
        ),
        "forecast": charts.forecast_chart(btc_price, result),
        "backtest": charts.backtest_chart(result),
        "sentiment": charts.sentiment_chart(sample_data.fear_greed(365)),
        "onchain": charts.onchain_chart(sample_data.onchain_btc(365)),
    }


class TestEveryChartBuilds:
    def test_all_twelve_render(self, figures):
        assert len(figures) == 12
        for name, fig in figures.items():
            assert fig is not None, name
            assert len(fig.data) > 0, f"{name} produced no traces"

    def test_every_chart_has_a_title(self, figures):
        for name, fig in figures.items():
            assert fig.layout.title.text, f"{name} has no title"


class TestPaletteIsClosed:
    def test_no_colour_outside_the_scheme(self, figures, allowed):
        """
        CHANGES.md section 11 verified this by dumping all twelve figures to
        JSON and extracting every hex value. That check now runs on every commit.
        """
        offenders = {}
        for name, fig in figures.items():
            extra = _hex_values(fig) - allowed
            if extra:
                offenders[name] = sorted(extra)
        assert not offenders, f"colours outside the palette: {offenders}"

    def test_no_plotly_stock_template_colourway(self, figures):
        """
        The Plasma ramp and the ten-hue qualitative set are the specific things
        that leaked in before the custom template was built from scratch.
        """
        plotly_defaults = {"#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#0D0887", "#F0F921"}
        for name, fig in figures.items():
            assert not (_hex_values(fig) & plotly_defaults), f"{name} pulled a stock colour"

    def test_phase_ribbon_uses_configured_phase_colours(self, figures):
        used = _hex_values(figures["phase_ribbon"])
        assert used & {c.upper() for c in config.PHASE_COLORS.values()}


class TestChartDataIntegrity:
    def test_forecast_chart_shows_history_and_projection(self, figures):
        """Three traces minimum: actuals, forecast line, and the interval band."""
        assert len(figures["forecast"].data) >= 3

    def test_backtest_chart_plots_predicted_against_actual(self, figures):
        assert len(figures["backtest"].data) >= 2

    def test_onchain_uses_a_secondary_axis(self, figures):
        """
        CHANGES.md section 7: the two series differ by roughly 3x, so sharing one
        axis flattened the smaller one.
        """
        fig = figures["onchain"]
        assert any(getattr(trace, "yaxis", None) == "y2" for trace in fig.data)

    def test_heatmap_uses_the_diverging_scale(self, figures):
        """
        px.imshow puts the scale on layout.coloraxis rather than on the trace.
        The midpoint is the page colour itself, so zero correlation fades into
        the background while strong relationships come forward.
        """
        fig = figures["correlation_heatmap"]
        scale = fig.layout.coloraxis.colorscale
        assert scale is not None
        midpoint = [c for pos, c in scale if pos == pytest.approx(0.5)]
        assert midpoint and midpoint[0].upper() == config.PAGE_BG.upper()
