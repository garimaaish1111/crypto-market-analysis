"""
The four-dataset explorer.

This tab exists to answer one question directly: *is each data source actually
working?* Every other tab shows a conclusion drawn from the data. This one shows
the data itself — where each feed came from, whether the last fetch was real or
generated, how many rows arrived, what date range they cover, what preprocessing
did to them, and the raw table underneath.

Each of the four providers gets its own panel with a live **Test connection**
button, so the answer is demonstrable on the spot rather than asserted.

``src/data/health.py`` is owned by the other half of this project and may not
exist yet. The import below is deliberately defensive: without it the tab still
renders everything it can from data already loaded, and only the connection
buttons are disabled. Being blocked on someone else's file is not a good enough
reason for a tab not to work.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from src.data import fx

try:  # pragma: no cover - exercised only once health.py lands
    from src.data import health as _health

    HEALTH_AVAILABLE = hasattr(_health, "check_feed") and hasattr(_health, "check_all")
except Exception:  # noqa: BLE001 - a missing or half-written module must not break the tab
    _health = None
    HEALTH_AVAILABLE = False


# Maps each provider panel to the pipeline source label and the frame it
# produced, so one loop can render all four consistently.
FEED_PANELS = [
    {
        "key": "coingecko",
        "provider": "CoinGecko",
        "dataset": "Crypto prices",
        "source_label": "Crypto (CoinGecko)",
        "what": "Daily price, volume and market cap for six digital assets, "
                f"fetched directly in {config.CURRENCY_LABEL}.",
        "endpoint": "api.coingecko.com/api/v3/coins/{id}/market_chart",
    },
    {
        "key": "yahoo",
        "provider": "Yahoo Finance",
        "dataset": "Traditional assets",
        "source_label": "Macro (Yahoo Finance)",
        "what": "S&P 500, gold, the US dollar index, the 10-year yield and crude "
                "oil — the comparison set for the correlation analysis.",
        "endpoint": "yfinance → query*.finance.yahoo.com",
    },
    {
        "key": "fng",
        "provider": "alternative.me",
        "dataset": "Market sentiment",
        "source_label": "Sentiment (Fear & Greed)",
        "what": "The crypto Fear & Greed index: a 0–100 composite of volatility, "
                "momentum, social media, surveys and Bitcoin dominance.",
        "endpoint": "api.alternative.me/fng/",
    },
    {
        "key": "blockchain",
        "provider": "Blockchain.info",
        "dataset": "Bitcoin on-chain",
        "source_label": "On-chain (Blockchain.info)",
        "what": "Daily transaction count and active addresses — how much the "
                "network is actually being used, independent of price.",
        "endpoint": "api.blockchain.info/charts/{metric}",
    },
]

_BADGE = {
    "live": ("🟢", "Live", "Fetched successfully from the provider."),
    "simulated": ("🔵", "Simulated", "Generated deliberately — CRYPTO_MODE is set to simulated."),
    "mixed": ("🟡", "Mixed", "Some assets fetched, others fell back to generated data."),
    "sample": ("🟠", "Sample", "The live call failed; generated data is standing in."),
    "n/a": ("⚪", "Not applicable", "No conversion needed in this currency."),
}


def _frame_for(panel: dict, data) -> pd.DataFrame:
    """The cleaned frame each provider produced, for preview and counting."""
    key = panel["key"]
    if key == "coingecko":
        return data.crypto_prices
    if key == "yahoo":
        return data.macro_prices
    if key == "fng":
        return data.sentiment
    if key == "blockchain":
        return data.onchain
    return pd.DataFrame()


def _reports_for(panel: dict, data) -> list:
    """Cleaning reports belonging to this provider."""
    prefix = {
        "coingecko": "Crypto",
        "yahoo": "Macro",
        "fng": "Sentiment",
        "blockchain": "On-chain",
    }[panel["key"]]
    return [r for r in data.reports if r.dataset.startswith(prefix)]


def _render_connection_control(panel: dict) -> None:
    """The Test connection button, or an explanation of why it is unavailable."""
    if not HEALTH_AVAILABLE:
        st.button(
            "Test connection",
            key=f"test_{panel['key']}",
            disabled=True,
            help="Connection testing lands with src/data/health.py.",
            width="stretch",
        )
        return

    if st.button("Test connection", key=f"test_{panel['key']}", width="stretch"):
        with st.spinner(f"Contacting {panel['provider']}…"):
            try:
                result = _health.check_feed(panel["key"])
            except Exception as exc:  # noqa: BLE001 - contract says never raise; trust nothing
                st.error(f"Health check itself failed: {exc}")
                return

        latency = getattr(result, "latency_ms", float("nan"))
        suffix = "" if pd.isna(latency) else f" · {latency:,.0f} ms"
        if getattr(result, "ok", False):
            st.success(f"{result.message}{suffix}")
        else:
            st.error(f"{result.message}{suffix}")


def _render_panel(panel: dict, data, days: int) -> None:
    """One provider: status, what it supplies, preview, and cleaning detail."""
    source = data.sources.get(panel["source_label"], "unknown")
    icon, label, meaning = _BADGE.get(source, ("⚪", source.title(), ""))

    frame = _frame_for(panel, data)
    reports = _reports_for(panel, data)

    st.markdown(f"### {icon} {panel['dataset']} — {panel['provider']}")
    st.caption(panel["what"])

    left, right = st.columns([3, 1])
    with left:
        rows = len(frame)
        if rows:
            start, end = frame.index.min(), frame.index.max()
            span = f"{start:%d %b %Y} → {end:%d %b %Y}"
        else:
            span = "—"

        stat = st.columns(4)
        stat[0].metric("Status", label)
        stat[1].metric("Rows", f"{rows:,}")
        stat[2].metric("Columns", f"{len(frame.columns):,}")
        stat[3].metric("Requested window", f"{days} days")
        st.caption(f"Covering **{span}**. {meaning}")
    with right:
        _render_connection_control(panel)

    if frame.empty:
        st.warning("This feed produced no usable rows, so nothing can be previewed.")
        st.markdown("---")
        return

    preview, quality = st.tabs(["Data preview", "Preprocessing"])

    with preview:
        st.dataframe(frame.tail(15), width="stretch")
        st.caption(
            f"Last 15 of {len(frame):,} rows, after cleaning. "
            "Values are exactly what the analysis tabs compute on."
        )
        csv = frame.to_csv().encode("utf-8")
        st.download_button(
            "Download this dataset (CSV)",
            data=csv,
            file_name=f"{panel['key']}_{days}d.csv",
            mime="text/csv",
            key=f"dl_{panel['key']}",
        )

    with quality:
        if not reports:
            st.info("No cleaning report was recorded for this feed.")
        else:
            st.dataframe(
                pd.DataFrame([r.as_dict() for r in reports]).set_index("Dataset"),
                width="stretch",
            )
            for report in reports:
                st.caption(report.summary())

    st.caption(f"Endpoint: `{panel['endpoint']}`")
    st.markdown("---")


def render(data, days: int) -> None:
    """
    Draw the whole tab.

    ``data`` is the ``MarketData`` bundle; ``days`` is the history window the
    sidebar asked for, shown so the reader can tell a short window from a
    truncated feed.
    """
    st.subheader("The four data sources")
    st.markdown(
        "Every number in this dashboard comes from one of the four feeds below. "
        "Each panel shows whether the last fetch was real or generated, what "
        "arrived, and what preprocessing did to it — and the **Test connection** "
        "button makes a fresh request so the answer is demonstrable rather than "
        "asserted."
    )

    live = sum(1 for p in FEED_PANELS if data.sources.get(p["source_label"]) == "live")
    summary = st.columns(4)
    summary[0].metric("Feeds live", f"{live} of {len(FEED_PANELS)}")
    summary[1].metric("Datasets cleaned", f"{len(data.reports)}")
    summary[2].metric(
        "Rows loaded",
        f"{sum(len(_frame_for(p, data)) for p in FEED_PANELS):,}",
    )
    summary[3].metric(
        f"USD/{config.CURRENCY_LABEL}",
        "—" if config.CURRENCY == "usd" else f"{data.latest_fx:,.2f}",
    )

    if data.unhealthy:
        st.warning(
            "Flagged by the data-quality check: **"
            + ", ".join(data.unhealthy)
            + "**. Open the Preprocessing tab in that panel for the detail."
        )

    if not HEALTH_AVAILABLE:
        st.info(
            "Per-feed connection testing arrives with `src/data/health.py`. "
            "Everything else on this tab reflects the data actually loaded."
        )

    st.markdown("---")
    for panel in FEED_PANELS:
        _render_panel(panel, data, days)

    with st.expander("How preprocessing treats each feed"):
        st.markdown(
            f"""
Cleaning happens in one place (`src/data/preprocessing.py`) rather than as
scattered `dropna` calls, and every step reports what it did.

| Step | Why it is in this order |
|---|---|
| Sort and de-duplicate the index | A repeated timestamp double-counts a day in every statistic below |
| Coerce to numeric | alternative.me returns its value as a *string*, which compares without raising |
| Collapse to one row per day | CoinGecko returns intraday points |
| Drop impossible values | A zero price becomes an infinity the moment a return is taken |
| Forward-fill gaps | Forward only — back-filling would carry a later value into an earlier day, which is look-ahead |
| Flag outliers, never delete them | VaR and CVaR exist to describe the tail; trimming it first would report a comfortable number that is wrong |

Traditional assets are **not** resampled onto a daily grid: those markets are
genuinely shut at weekends, and inventing rows would manufacture zero-return days
that drag every measured correlation toward zero.

Prices are denominated in **{config.CURRENCY_LABEL}**. Crypto is fetched from
CoinGecko directly in {config.CURRENCY_LABEL}, so no conversion is applied to it.
Dollar-quoted traditional assets are converted at the daily USD/{config.CURRENCY_LABEL}
rate; the US Dollar Index and the 10-year yield are left alone, because an index
level and a rate in percentage points are not prices in dollars.
            """
        )
