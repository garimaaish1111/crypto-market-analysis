"""
Plotly chart builders used by the Streamlit dashboard.

Keeping figure construction here means the dashboard file stays about layout and
the charts remain reusable and testable on their own.

Colour directs attention rather than decorating, and the whole interface runs on
five values defined in ``config``:

    INK          #212121   observed data — price lines, actuals, bear phase
    ACCENT       #00BCD4   whatever the tab wants you to look at
    ACCENT_DARK  #0097A7   emphasis inside the accent family
    MUTED        #607D8B   context that should recede
    MUTED_LIGHT  #90A4AE   tertiary series and base lines

The two tints appear only as hairline gridlines and as the neutral midpoint of
the correlation scale. Change any value in ``config`` and every chart follows.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import config
from src.analysis import cycles, volatility

PHASE_COLORS = config.PHASE_COLORS

# A named template built once at import.
#
# Plotly's stock templates embed their own colourways and per-trace-type colour
# scales in every figure they produce — the Plasma ramp for heatmaps, a ten-hue
# qualitative set for lines — so a trace that forgets to set a colour silently
# picks up an out-of-scheme hue. Rather than inherit and override, the template
# below starts empty and declares only what this dashboard uses. Nothing outside
# the five colours can reach the screen, whether or not a future chart remembers
# to be explicit. Axis, font and background styling is applied per figure in
# ``_base_layout``.
_SEQUENTIAL = [
    [0.0, config.PAGE_BG],
    [0.35, config.MUTED],
    [0.7, config.ACCENT],
    [1.0, config.ACCENT_DARK],
]

_COLORWAY = [
    config.SERIES_PRIMARY,
    config.ACCENT,
    config.MUTED,
    config.ACCENT_DARK,
    config.MUTED_LIGHT,
]

_template = go.layout.Template()
_template.layout.colorway = _COLORWAY
_template.layout.piecolorway = _COLORWAY
_template.layout.font = dict(color=config.TEXT, family="system-ui, sans-serif")
_template.layout.paper_bgcolor = "rgba(0,0,0,0)"
_template.layout.plot_bgcolor = "rgba(0,0,0,0)"
_template.layout.colorscale.sequential = _SEQUENTIAL
_template.layout.colorscale.sequentialminus = _SEQUENTIAL
_template.layout.colorscale.diverging = config.CORRELATION_SCALE
_template.layout.xaxis = dict(
    gridcolor=config.GRID, linecolor=config.GRID, zerolinecolor=config.GRID, ticks="outside"
)
_template.layout.yaxis = dict(
    gridcolor=config.GRID, linecolor=config.GRID, zerolinecolor=config.GRID, ticks="outside"
)
_template.data.heatmap = [go.Heatmap(colorscale=config.CORRELATION_SCALE)]
_template.data.bar = [go.Bar(marker=dict(color=config.SERIES_TERTIARY))]

pio.templates["crypto"] = _template
pio.templates.default = "crypto"


def _base_layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(color=config.TEXT, size=16)),
        height=height,
        margin=dict(l=40, r=20, t=54, b=30),
        template="crypto",
        hovermode="x unified",
        font=dict(color=config.TEXT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(gridcolor=config.GRID, zerolinecolor=config.GRID, linecolor=config.GRID)
    fig.update_yaxes(gridcolor=config.GRID, zerolinecolor=config.GRID, linecolor=config.GRID)
    return fig


def price_with_ma(price: pd.Series, symbol: str) -> go.Figure:
    """Price with its adaptive short and long moving averages."""
    short, long = config.ma_windows(len(price))
    ma = cycles.moving_averages(price, short, long)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=ma.index, y=ma["price"], name=symbol, line=dict(color=config.SERIES_PRIMARY, width=1.8))
    )
    fig.add_trace(
        go.Scatter(
            x=ma.index,
            y=ma["ma_short"],
            name=f"{short}-day MA",
            line=dict(color=config.ACCENT, width=1.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ma.index,
            y=ma["ma_long"],
            name=f"{long}-day MA",
            line=dict(color=config.MUTED, width=1.2, dash="dash"),
        )
    )
    return _base_layout(fig, f"{symbol} price and moving averages")


def volume_bars(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=df.index, y=df["volume"], marker_color=config.SERIES_TERTIARY))
    return _base_layout(fig, f"{symbol} trading volume", height=260)


def rolling_vol_chart(price: pd.Series, symbol: str) -> go.Figure:
    rv = volatility.rolling_volatility(price)
    fig = go.Figure(
        go.Scatter(
            x=rv.index,
            y=rv,
            line=dict(color=config.ACCENT, width=2),
            fill="tozeroy",
            fillcolor=config.FILL_ACCENT,
            name=f"{config.ROLLING_VOL_WINDOW}-day annualised vol",
        )
    )
    if not rv.empty:
        fig.add_hline(
            y=float(rv.mean()),
            line_dash="dot",
            line_color=config.MUTED,
            annotation_text="window average",
            annotation_font_color=config.MUTED,
        )
    fig.update_yaxes(tickformat=".0%")
    return _base_layout(fig, f"{symbol} rolling volatility (annualised)", height=300)


def drawdown_chart(price: pd.Series, symbol: str) -> go.Figure:
    dd = volatility.drawdown_series(price)
    fig = go.Figure(
        go.Scatter(
            x=dd.index,
            y=dd,
            fill="tozeroy",
            line=dict(color=config.MUTED, width=1.4),
            fillcolor=config.FILL_MUTED,
            name="Drawdown",
        )
    )
    fig.update_yaxes(tickformat=".0%")
    return _base_layout(fig, f"{symbol} drawdown from window peak", height=300)


def phase_ribbon(price: pd.Series, symbol: str) -> go.Figure:
    """
    Price line with each classified day marked by its inferred cycle phase.

    Days inside the moving-average warm-up carry no marker at all. Giving them a
    sixth colour would imply a sixth phase; leaving them as the bare line reads
    correctly as "not yet classifiable" and keeps the palette closed at five.
    """
    phases = cycles.phase_series(price)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=price.index,
            y=price,
            line=dict(color=config.SERIES_TERTIARY, width=1),
            name="Price",
            hoverinfo="skip",
        )
    )
    for phase, color in PHASE_COLORS.items():
        mask = (phases == phase).to_numpy()
        if mask.any():
            fig.add_trace(
                go.Scatter(
                    x=price.index[mask],
                    y=price[mask],
                    mode="markers",
                    marker=dict(color=color, size=5),
                    name=phase,
                )
            )
    return _base_layout(fig, f"{symbol} market-cycle phases", height=420)


def phase_distribution(phases: pd.Series) -> go.Figure:
    """How many days the window spent in each phase."""
    counts = phases.value_counts()
    fig = go.Figure(
        go.Bar(
            x=counts.index,
            y=counts.to_numpy(),
            marker_color=[PHASE_COLORS.get(p, config.UNDETERMINED_COLOR) for p in counts.index],
            text=counts.to_numpy(),
            textposition="outside",
            textfont=dict(color=config.TEXT),
        )
    )
    return _base_layout(fig, "Days spent in each phase", height=300)


def correlation_heatmap(corr: pd.DataFrame, title: str = "Correlation matrix (daily returns)") -> go.Figure:
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale=config.CORRELATION_SCALE,
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig.update_traces(textfont=dict(size=11, color=config.CELL_TEXT))
    return _base_layout(fig, title, height=520)


def rolling_corr_chart(series: pd.Series, a: str, b: str) -> go.Figure:
    fig = go.Figure(
        go.Scatter(x=series.index, y=series, line=dict(color=config.ACCENT, width=2), name=f"{a} vs {b}")
    )
    fig.add_hline(y=0, line_dash="dot", line_color=config.MUTED)
    fig.update_yaxes(range=[-1, 1])
    return _base_layout(fig, f"Rolling correlation: {a} vs {b}", height=320)


def forecast_chart(price: pd.Series, result, lookback: int = 120) -> go.Figure:
    """Recent history plus the forward forecast and its uncertainty band."""
    recent = price.iloc[-lookback:]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=recent.index, y=recent, name="History", line=dict(color=config.SERIES_PRIMARY, width=1.8))
    )
    fig.add_trace(
        go.Scatter(
            x=list(result.upper.index) + list(result.lower.index[::-1]),
            y=list(result.upper.to_numpy()) + list(result.lower.to_numpy()[::-1]),
            fill="toself",
            fillcolor=config.FILL_ACCENT,
            line=dict(color="rgba(0,0,0,0)"),
            name="80% interval",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.forecast.index,
            y=result.forecast,
            name="Forecast",
            line=dict(color=config.ACCENT_DARK, width=2, dash="dash"),
        )
    )
    return _base_layout(fig, f"{result.symbol} {len(result.forecast)}-day ARIMA forecast", height=420)


def backtest_chart(result) -> go.Figure:
    """Held-out period: what actually happened against the one-step-ahead prediction."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.test_actual.index,
            y=result.test_actual,
            name="Actual",
            line=dict(color=config.SERIES_PRIMARY, width=1.8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.fitted_test.index,
            y=result.fitted_test,
            name="Predicted (1 day ahead)",
            line=dict(color=config.ACCENT, width=1.6, dash="dot"),
        )
    )
    return _base_layout(fig, f"{result.symbol} walk-forward validation", height=320)


def sentiment_chart(sentiment: pd.DataFrame) -> go.Figure:
    """Fear & Greed with its sentiment bands."""
    fig = go.Figure(
        go.Scatter(
            x=sentiment.index,
            y=sentiment["fng_value"],
            line=dict(color=config.SERIES_PRIMARY, width=1.6),
            name="Fear & Greed",
        )
    )
    # Fear at the bottom in charcoal, greed at the top in cyan.
    fig.add_hrect(y0=0, y1=25, fillcolor=config.MUTED, opacity=0.30, line_width=0)
    fig.add_hrect(y0=25, y1=45, fillcolor=config.MUTED, opacity=0.15, line_width=0)
    fig.add_hrect(y0=55, y1=75, fillcolor=config.ACCENT, opacity=0.14, line_width=0)
    fig.add_hrect(y0=75, y1=100, fillcolor=config.ACCENT_DARK, opacity=0.20, line_width=0)
    fig.update_yaxes(range=[0, 100])
    return _base_layout(fig, "Crypto Fear & Greed index", height=320)


def onchain_chart(onchain: pd.DataFrame) -> go.Figure:
    """
    Bitcoin transaction count and active addresses on separate axes.

    The two series differ by roughly a factor of three, so plotting them on one
    axis flattens the smaller of them into a straight line.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=onchain.index,
            y=onchain["tx_count"],
            name="Daily transactions",
            line=dict(color=config.SERIES_PRIMARY, width=1.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=onchain.index,
            y=onchain["active_addresses"],
            name="Active addresses",
            line=dict(color=config.ACCENT, width=1.6),
            yaxis="y2",
        )
    )
    fig.update_layout(
        yaxis=dict(title=dict(text="Transactions", font=dict(color=config.TEXT))),
        yaxis2=dict(
            title=dict(text="Active addresses", font=dict(color=config.ACCENT_DARK)),
            overlaying="y",
            side="right",
            showgrid=False,
        ),
    )
    return _base_layout(fig, "Bitcoin network activity", height=340)
