"""
Cryptocurrency Market Analysis System — Streamlit dashboard.

Run with:  streamlit run app.py

Tabs:
  Overview            — snapshot prices, sentiment, risk and cycle summary tables
  Volatility & Risk   — rolling vol, drawdown, VaR/CVaR, Sharpe/Sortino
  Market Cycles       — phase ribbon, RSI, moving averages
  Correlation         — crypto vs traditional assets, including a stress split
  Forecast            — ARIMA forecast, walk-forward validation, naive baseline
  Data & Sources      — feed status, sentiment, on-chain metrics, raw tables
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from src import pipeline
from src.analysis import correlation, cycles, forecast, volatility
from src.data import cache
from src.viz import charts

st.set_page_config(page_title="Crypto Market Analysis System", layout="wide")


# --------------------------------------------------------------------------- #
# Startup consistency check
#
# The modules in this project are versioned together: charts.py reads theme
# tokens from config, crypto.py reads CRYPTO_MODE, pipeline unpacks a per-coin
# source map. Replacing some files but not others produces a bare AttributeError
# deep inside a cached call, which says nothing useful about the cause. This
# names the stale file instead.
# --------------------------------------------------------------------------- #
_REQUIRED_CONFIG = {
    "CRYPTO_MODE": "live-vs-simulated switch",
    "THEME": "dark/light switch",
    "PAGE_BG": "theme colour tokens",
    "TEXT": "theme colour tokens",
    "SERIES_PRIMARY": "theme colour tokens",
    "PHASE_COLORS": "market-cycle palette",
    "ma_windows": "adaptive moving-average helper",
    "coingecko_base": "CoinGecko host selection",
}

_missing = [name for name in _REQUIRED_CONFIG if not hasattr(config, name)]
if _missing:
    st.error(
        "**config.py is out of date.**\n\n"
        "The rest of the project expects settings this copy of `config.py` does not "
        "define, which means some files were replaced and others were not:\n\n"
        + "\n".join(f"- `{name}` — {_REQUIRED_CONFIG[name]}" for name in _missing)
        + "\n\nCopy `config.py` from the distribution folder over the one in this "
        "project, then reload. Replacing the whole folder at once avoids this."
    )
    st.stop()

# Light styling to carry the three-colour scheme into Streamlit's own chrome,
# which Plotly theming does not reach.
st.markdown(
    f"""
    <style>
      /* Page and panels */
      .stApp {{ background: {config.PAGE_BG}; color: {config.TEXT}; }}
      section[data-testid="stSidebar"] {{
        background: {config.PANEL_BG};
        border-right: 1px solid {config.BORDER};
      }}
      h1, h2, h3, h4, p, li, label, span, div {{ color: {config.TEXT}; }}
      hr {{ border-color: {config.BORDER}; }}

      /* Metrics */
      [data-testid="stMetricValue"] {{ color: {config.TEXT}; }}
      [data-testid="stMetricLabel"] {{ color: {config.TEXT_MUTED}; }}

      /* Streamlit colours metric deltas red and green by default. The arrow
         already carries the direction, so the colour only adds a hue that is
         not in this scheme. */
      [data-testid="stMetricDelta"] {{ color: {config.TEXT_MUTED} !important; }}
      [data-testid="stMetricDelta"] svg {{ fill: {config.TEXT_MUTED} !important; }}

      /* Alerts ship as yellow, blue and green boxes. Same treatment: one
         neutral panel with a cyan marker. */
      .stAlert {{
        background: {config.FILL_MUTED} !important;
        border: 1px solid {config.BORDER} !important;
        border-radius: 6px;
      }}
      .stAlert p, .stAlert div, .stAlert li {{ color: {config.TEXT} !important; }}
      .stAlert svg {{ fill: {config.ACCENT} !important; }}

      /* Tabs, inputs, tables */
      .stTabs [aria-selected="true"] {{ color: {config.ACCENT} !important; }}
      .stTabs [data-baseweb="tab-highlight"] {{ background: {config.ACCENT} !important; }}
      [data-testid="stExpander"] {{ border: 1px solid {config.BORDER}; border-radius: 6px; }}
      [data-testid="stDataFrame"] {{ border: 1px solid {config.BORDER}; border-radius: 6px; }}
      .stSlider [data-baseweb="slider"] div[role="slider"] {{ background: {config.ACCENT} !important; }}
      code {{ color: {config.ACCENT} !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading market data…", ttl=60 * 60)
def get_data(days: int) -> pipeline.MarketData:
    return pipeline.load_market_data(days)


@st.cache_data(show_spinner="Fitting and validating ARIMA…", ttl=60 * 60)
def get_forecast(_price: pd.Series, symbol: str, horizon: int, days: int) -> forecast.ForecastResult:
    """
    ``_price`` is excluded from the cache key by Streamlit's underscore
    convention, so ``days`` has to be passed explicitly. Without it, changing the
    history window would return the forecast computed on the previous window.
    """
    return forecast.forecast_price(_price, symbol, horizon=horizon)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("Crypto Market Analysis")
st.sidebar.caption("Risk, market cycles, correlation and forecasting")

days = st.sidebar.select_slider(
    "History window (days)",
    options=list(config.HISTORY_WINDOWS),
    value=config.DEFAULT_DAYS,
)

if st.sidebar.button("Refresh data", width="stretch"):
    removed = cache.clear_all()      # clear the on-disk cache…
    st.cache_data.clear()            # …and Streamlit's in-memory cache
    st.toast(f"Cleared {removed} cached responses. Refetching.")
    st.rerun()

data = get_data(days)
symbol = st.sidebar.selectbox("Primary asset", data.symbols, index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Data sources")
for feed, source in data.sources.items():
    icon = {"live": "🟢", "simulated": "🔵", "mixed": "🟡", "sample": "🟠"}.get(source, "⚪")
    st.sidebar.write(f"{icon} {feed}: **{source}**")

# A deliberate mode and an unplanned failure produce the same data but mean
# different things, so they are surfaced differently: the first is stated once
# and calmly, the second is flagged.
simulated_by_choice = {f: s for f, s in data.sources.items() if s == "simulated"}
unexpected = {f: s for f, s in data.sources.items() if s in ("mixed", "sample")}

# Convenient handles
crypto_df = data.crypto_frames[symbol]
price = crypto_df["price"]
ma_short, ma_long = config.ma_windows(len(price))


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("Cryptocurrency Market Analysis System")
st.markdown(
    "Volatility and risk, market-cycle identification, crypto-versus-traditional "
    "correlation, and time-series forecasting — built to support data-driven "
    "investment decisions."
)

if unexpected:
    detail = ""
    if data.simulated_symbols:
        detail = (
            "  Affected coins: **" + ", ".join(data.simulated_symbols) + "**. "
            "Any table comparing these against live assets is mixing measured and "
            "generated data."
        )
    st.warning(
        f"**A feed dropped unexpectedly** ({', '.join(unexpected)}). "
        "Generated data is standing in so every chart still renders. "
        "Use **Refresh data** once you have a connection." + detail
    )
elif simulated_by_choice:
    st.caption(
        "Crypto price series are generated rather than fetched — CoinGecko's free tier "
        "rate limits below the six requests this dashboard needs. The generator carries "
        "a realistic cross-asset correlation structure, so every metric below is computed "
        "by the same code on the same shape of data. Set `CRYPTO_MODE = \"live\"` in "
        "`config.py` to fetch instead. Macro, sentiment and on-chain feeds are live."
    )

tabs = st.tabs(
    ["Overview", "Volatility & Risk", "Market Cycles", "Correlation", "Forecast", "Data & Sources"]
)

# --------------------------------------------------------------------------- #
# Tab 1 — Overview
# --------------------------------------------------------------------------- #
with tabs[0]:
    latest = price.iloc[-1]
    change_24h = price.pct_change().iloc[-1]
    change_30d = price.iloc[-1] / price.iloc[-min(31, len(price))] - 1
    fng_latest = data.sentiment["fng_value"].iloc[-1]
    fng_label = data.sentiment["fng_label"].iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{symbol} price", f"${latest:,.2f}", f"{change_24h:+.2%} (24h)")
    c2.metric("30-day change", f"{change_30d:+.1%}")
    c3.metric("Annualised volatility", f"{volatility.annualised_volatility(price):.0%}")
    c4.metric("Fear & Greed", f"{fng_latest:.0f}", fng_label)

    st.plotly_chart(charts.price_with_ma(price, symbol), width="stretch")
    st.plotly_chart(charts.volume_bars(crypto_df, symbol), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Risk snapshot")
        risk_rows = [
            volatility.risk_profile(s, data.crypto_frames[s]["price"]).as_dict()
            for s in data.symbols
        ]
        st.dataframe(pd.DataFrame(risk_rows).set_index("Symbol"), width="stretch")
    with right:
        st.subheader("Cycle snapshot")
        cycle_rows = [
            cycles.current_phase(s, data.crypto_frames[s]["price"]).as_dict()
            for s in data.symbols
        ]
        st.dataframe(pd.DataFrame(cycle_rows).set_index("Symbol"), width="stretch")

# --------------------------------------------------------------------------- #
# Tab 2 — Volatility & Risk
# --------------------------------------------------------------------------- #
with tabs[1]:
    st.subheader(f"Volatility and risk — {symbol}")
    profile = volatility.risk_profile(symbol, price)
    formatted = profile.as_dict()

    m = st.columns(5)
    m[0].metric("VaR 95% (1-day)", formatted["VaR 95%"])
    m[1].metric("CVaR 95%", formatted["CVaR 95%"])
    m[2].metric("Max drawdown", formatted["Max Drawdown"])
    m[3].metric("Sharpe / Sortino", f"{formatted['Sharpe']} / {formatted['Sortino']}")
    m[4].metric("Volatility regime", profile.regime)

    st.plotly_chart(charts.rolling_vol_chart(price, symbol), width="stretch")
    st.plotly_chart(charts.drawdown_chart(price, symbol), width="stretch")

    with st.expander("What these metrics mean"):
        st.markdown(
            f"""
- **VaR 95%** — the daily loss not exceeded on 95% of days. Estimated from the
  observed return distribution rather than a normal curve, because crypto returns
  have far fatter tails than a normal distribution allows for.
- **CVaR 95%** — the *average* loss on the worst 5% of days. VaR tells you where
  the threshold sits; CVaR tells you how bad it gets once you cross it.
- **Max drawdown** — the deepest peak-to-trough fall inside this {days}-day
  window. It is a window high, not an all-time high.
- **Sharpe / Sortino** — return per unit of risk. Sortino counts only shortfall
  against the {config.RISK_FREE_RATE:.0%} risk-free rate, so upside volatility is
  not treated as a problem.
- **Volatility regime** — where the current {config.ROLLING_VOL_WINDOW}-day
  volatility sits against the rest of this window.
            """
        )

# --------------------------------------------------------------------------- #
# Tab 3 — Market Cycles
# --------------------------------------------------------------------------- #
with tabs[2]:
    st.subheader(f"Market-cycle phase — {symbol}")
    phase = cycles.current_phase(symbol, price)
    phase_formatted = phase.as_dict()

    m = st.columns(4)
    m[0].metric("Current phase", phase.phase)
    m[1].metric(f"RSI ({config.RSI_WINDOW})", phase_formatted[f"RSI({config.RSI_WINDOW})"])
    m[2].metric(f"Price vs {ma_long}-day MA", phase_formatted[f"Price vs {ma_long}MA"])
    m[3].metric("Drawdown vs window high", phase_formatted["Drawdown vs window high"])

    if (ma_short, ma_long) != (config.MA_SHORT, config.MA_LONG):
        st.info(
            f"This {days}-day window is too short for the standard "
            f"{config.MA_SHORT}/{config.MA_LONG}-day averages, so **{ma_short}/{ma_long}** "
            "are in use instead. Phases identified on shorter averages react faster "
            "and turn over more often. Select 365 days for the standard pair.",
        )

    phases = cycles.phase_series(price)
    st.plotly_chart(charts.phase_ribbon(price, symbol), width="stretch")
    st.plotly_chart(charts.phase_distribution(phases), width="stretch")

    undetermined = int((phases == "Undetermined").sum())
    if undetermined:
        st.caption(
            f"The first {undetermined} days carry no label: a {ma_long}-day average "
            f"needs {ma_long} observations before it produces its first value."
        )

    with st.expander("How phases are identified"):
        st.markdown(
            f"""
Phases follow the Wyckoff cycle — **Accumulation → Markup → Distribution →
Markdown** — inferred from three signals, so every label is traceable:

- **Trend structure**: an uptrend is price > {ma_short}-day MA > {ma_long}-day MA;
  a downtrend is the same stack inverted.
- **Momentum**: RSI-{config.RSI_WINDOW}, Wilder's smoothing. Above
  {cycles.RSI_STRONG} confirms an advance, below {cycles.RSI_WEAK} confirms a decline.
- **Drawdown**: distance below the running peak.

| Trend | Momentum / drawdown | Phase |
|---|---|---|
| Up | RSI ≥ {cycles.RSI_STRONG} | Markup (Bull) |
| Up | RSI < {cycles.RSI_STRONG} and within {abs(cycles.NEAR_HIGHS):.0%} of the peak | Distribution |
| Down | RSI ≤ {cycles.RSI_WEAK} and drawdown < {cycles.DEEP_DRAWDOWN:.0%} | Markdown (Bear) |
| Down | RSI recovering, drawdown < {cycles.DEEP_DRAWDOWN:.0%} | Accumulation |
| Down | otherwise | Markdown (Bear) |
| Sideways | drawdown < {cycles.BASING_DRAWDOWN:.0%} and RSI < 50 | Accumulation |
| Sideways | within {abs(cycles.NEAR_HIGHS):.0%} of the peak and RSI > {cycles.RSI_STRONG} | Distribution |
| Sideways | otherwise | Transition |

One caveat worth keeping in view: a full crypto cycle historically runs about
four years, so a {days}-day window shows a slice of one rather than the whole
shape. Read these labels as a description of the current regime, not a position
on a complete cycle.
            """
        )

# --------------------------------------------------------------------------- #
# Tab 4 — Correlation
# --------------------------------------------------------------------------- #
with tabs[3]:
    st.subheader("Digital versus traditional assets")
    returns = correlation.align_returns(data.crypto_prices, data.macro_prices)

    st.caption(
        f"Aligned on {len(returns)} common trading days. Crypto trades every day and "
        "traditional markets do not, so both sides are differenced only on dates they "
        "share — a Friday-to-Monday move spans the same three days for each."
    )

    # A rate-limited provider returns the column empty rather than failing, and
    # correlation needs every column on the same day. Dropping the dead asset
    # keeps the rest of the matrix usable, but the reader has to be told which
    # one is missing rather than silently seeing a smaller table.
    excluded = [c for c in data.macro_prices.columns if c not in returns.columns]
    if excluded:
        st.warning(
            f"Excluded from this analysis: **{', '.join(excluded)}** — the feed returned "
            "too few observations over this window to correlate. Every other asset below "
            "is unaffected. Try **Refresh data** in the sidebar to refetch."
        )

    method = st.radio(
        "Correlation method",
        ["pearson", "spearman"],
        horizontal=True,
        format_func=lambda m: "Pearson (linear)" if m == "pearson" else "Spearman (rank)",
        help="A large gap between the two means the Pearson figure is being driven by a few extreme days.",
    )

    st.plotly_chart(
        charts.correlation_heatmap(
            correlation.correlation_matrix(returns, method=method),
            title=f"Correlation matrix — {method.title()}, daily returns",
        ),
        width="stretch",
    )

    st.write("**Crypto to macro correlation, full window**")
    st.dataframe(
        correlation.crypto_macro_summary(
            returns, data.symbols, list(data.macro_prices.columns), method=method
        ),
        width="stretch",
    )

    st.markdown("---")
    st.write("**Does the diversification hold when markets fall?**")
    st.caption(
        f"Correlation measured separately on the worst {config.STRESS_QUANTILE:.0%} of "
        f"{config.STRESS_BENCHMARK} days and on everything else. A large positive change "
        "means crypto co-moves with equities precisely when a hedge would be most useful."
    )
    regime = correlation.regime_table(returns, data.symbols)
    if not regime.empty:
        st.dataframe(regime, width="stretch")

    st.markdown("---")
    st.write("**Rolling correlation**")
    # Offer only assets that survived alignment — listing an excluded one would
    # leave the user picking an option that silently renders nothing.
    comparable = [c for c in data.macro_prices.columns if c in returns.columns]
    if not comparable:
        st.info("No traditional asset has enough overlapping data to plot against right now.")
    else:
        rc1, rc2 = st.columns(2)
        macro_choice = rc1.selectbox("Compare against", comparable, index=0)
        window = rc2.select_slider(
            "Rolling window (days)", options=list(config.CORRELATION_WINDOWS), value=30
        )
        if symbol in returns.columns:
            series = correlation.rolling_correlation(returns, symbol, macro_choice, window)
            st.plotly_chart(charts.rolling_corr_chart(series, symbol, macro_choice), width="stretch")
            b = correlation.beta(returns, symbol, macro_choice)
            st.caption(
                f"β of {symbol} to {macro_choice}: **{b:.2f}**. Above 1 means {symbol} "
                f"amplifies {macro_choice}'s moves rather than cushioning them."
            )

# --------------------------------------------------------------------------- #
# Tab 5 — Forecast
# --------------------------------------------------------------------------- #
with tabs[4]:
    st.subheader(f"Time-series forecast — {symbol}")
    horizon = st.slider("Forecast horizon (days)", 7, 60, config.FORECAST_HORIZON, step=7)
    result = get_forecast(price, symbol, horizon, days)

    m = st.columns(4)
    m[0].metric("ARIMA order", str(result.order))
    m[1].metric("RMSE", f"${result.rmse:,.0f}", f"naive ${result.naive_rmse:,.0f}", delta_color="off")
    m[2].metric("MAPE", f"{result.mape:.2f}%", f"naive {result.naive_mape:.2f}%", delta_color="off")
    m[3].metric(
        "Skill vs random walk",
        f"{result.skill:+.1%}" if pd.notna(result.skill) else "—",
        "beats baseline" if result.beats_naive else "no better than baseline",
        delta_color="normal" if result.beats_naive else "off",
    )

    st.info(result.verdict())

    st.plotly_chart(charts.forecast_chart(price, result), width="stretch")
    st.plotly_chart(charts.backtest_chart(result), width="stretch")

    with st.expander("How this is validated"):
        st.markdown(
            f"""
1. The last {config.FORECAST_TEST_FRACTION:.0%} of history is held out.
2. Nine ARIMA orders — (p, 1, q) for p, q in 0–2 — are fitted on the training
   split and the lowest AIC wins. Differencing once on log-price means the model
   works on log returns, which is the stationary quantity.
3. Accuracy is measured **one day ahead at a time**, walking forward through the
   test period and revealing each actual observation before predicting the next.
   A single long forecast across the whole test period would mostly measure how
   far the price drifted, not how good the model is.
4. The same days are scored for the **naive random walk** — tomorrow's price is
   today's price. Skill is 1 − RMSE(model) ÷ RMSE(naive).
5. The chosen order is refitted on the full history to project ahead.

A skill score at or below zero is the expected result for crypto, not a bug: daily
returns are close to unpredictable, and a model that cannot beat "assume no change"
is telling you something true about the asset. Read the widening band as a range of
plausible outcomes rather than a price target.
            """
        )

# --------------------------------------------------------------------------- #
# Tab 6 — Data & Sources
# --------------------------------------------------------------------------- #
with tabs[5]:
    st.subheader("Sentiment")
    st.plotly_chart(charts.sentiment_chart(data.sentiment), width="stretch")
    st.caption(
        "The Fear & Greed index is a composite of volatility, momentum, social media "
        "activity, surveys and Bitcoin dominance. It stands in for direct social-media "
        "text analysis, which would need a paid data feed."
    )

    st.markdown("---")
    st.subheader("Bitcoin on-chain activity")
    oc1, oc2 = st.columns(2)
    oc1.metric("Daily transactions", f"{data.onchain['tx_count'].iloc[-1]:,.0f}")
    oc2.metric("Active addresses", f"{data.onchain['active_addresses'].iloc[-1]:,.0f}")
    st.plotly_chart(charts.onchain_chart(data.onchain), width="stretch")
    st.caption(
        "On-chain metrics measure the network rather than the market: how much the "
        "blockchain is actually being used, independent of what traders think it is "
        "worth. These two series are reported here but do not feed the risk, cycle or "
        "correlation models — extending them into the analysis is the clearest next step."
    )

    st.markdown("---")
    st.subheader("Feed status")
    st.dataframe(
        pd.DataFrame(
            {"Feed": list(data.sources), "Source": list(data.sources.values())}
        ).set_index("Feed"),
        width="stretch",
    )
    st.caption(
        f"{cache.entry_count()} responses currently cached, valid for "
        f"{config.CACHE_TTL_HOURS} hours. Crypto mode: **{config.CRYPTO_MODE}**."
    )

    with st.expander(f"Raw {symbol} price and volume"):
        st.dataframe(crypto_df.tail(50), width="stretch")

st.markdown("---")
st.caption(
    "Built for educational and analytical use. Not financial advice. "
    "Data: CoinGecko, Yahoo Finance, alternative.me, Blockchain.info, "
    "with a simulated fallback when a feed is unreachable."
)
