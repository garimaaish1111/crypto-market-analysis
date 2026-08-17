"""
Central configuration for the Cryptocurrency Market Analysis System.

Everything a user might reasonably want to change (which coins to track, which
macro assets to compare against, cache behaviour, risk parameters, the colour
palette) lives here so the rest of the codebase stays free of magic numbers.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Crypto assets (CoinGecko ids -> display symbol)
# --------------------------------------------------------------------------- #
CRYPTO_ASSETS: dict[str, str] = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "solana": "SOL",
    "binancecoin": "BNB",
    "ripple": "XRP",
    "cardano": "ADA",
}

# --------------------------------------------------------------------------- #
# Traditional / macro assets (Yahoo Finance tickers -> display name)
# Used for digital-vs-traditional correlation analysis.
# --------------------------------------------------------------------------- #
MACRO_ASSETS: dict[str, str] = {
    "^GSPC": "S&P 500",
    "GC=F": "Gold",
    "DX-Y.NYB": "US Dollar Index",
    "^TNX": "US 10Y Yield",
    "CL=F": "Crude Oil",
}

# Assets quoted as a *rate*, not a price. A move from 4.00% to 4.10% is a
# 10 basis-point change, not a "+2.5% return", so these columns are differenced
# rather than percent-changed before any correlation work.
YIELD_ASSETS: set[str] = {"US 10Y Yield"}

# The benchmark used for the risk-on / risk-off correlation split.
STRESS_BENCHMARK = "S&P 500"

# --------------------------------------------------------------------------- #
# Data / API settings
# --------------------------------------------------------------------------- #
# Live CoinGecko calls are currently disabled. The anonymous free tier rate
# limits at a handful of requests per minute, and six coins in a row reliably
# earns a 429 on the last few — which produced a stream of warnings and a risk
# table quietly mixing measured and generated assets.
#
# In "simulated" mode the loader does not touch the network at all: no request,
# no exception, no warning. Crypto series come from the deterministic generator
# in sample_data.py, which carries a realistic cross-asset correlation structure
# so every analysis still exercises the same code paths.
#
# Set CRYPTO_MODE = "live" to re-enable, ideally with a free demo key.
# Macro, sentiment and on-chain feeds are unaffected and remain live.
CRYPTO_MODE = "live"   # "live" or "simulated"

# CoinGecko free tier is heavily rate limited. A free "demo" key raises the
# ceiling substantially and takes two minutes to create at
# https://www.coingecko.com/en/developers/dashboard — set COINGECKO_API_KEY and
# leave COINGECKO_PLAN as "demo". Demo and Pro keys use different hosts and
# different header names, which is why the plan has to be declared.
COINGECKO_PLAN = os.getenv("COINGECKO_PLAN", "demo")   # "demo" or "pro"
COINGECKO_MIN_INTERVAL = 2.5    # seconds between live calls
COINGECKO_MAX_ATTEMPTS = 4      # retries before falling back to sample data
COINGECKO_BACKOFF = 3.0         # initial backoff, doubles each retry
COINGECKO_MAX_WAIT = 30.0       # cap on any single wait
FNG_URL = "https://api.alternative.me/fng/"               # crypto Fear & Greed
BLOCKCHAIN_CHARTS = "https://api.blockchain.info/charts"  # BTC on-chain metrics

DEFAULT_DAYS = 365            # history window (CoinGecko free tier max)
HISTORY_WINDOWS = (90, 180, 270, 365)
REQUEST_TIMEOUT = 20          # seconds
CACHE_TTL_HOURS = 6           # re-use cached responses within this window
API_KEY = os.getenv("COINGECKO_API_KEY", "")  # optional; blank = anonymous free tier


def coingecko_base() -> str:
    """Pro keys use a different host from demo and anonymous access."""
    if API_KEY and COINGECKO_PLAN == "pro":
        return "https://pro-api.coingecko.com/api/v3"
    return "https://api.coingecko.com/api/v3"

# --------------------------------------------------------------------------- #
# Analysis parameters
# --------------------------------------------------------------------------- #
RISK_FREE_RATE = 0.04         # annualised, for Sharpe / Sortino
TRADING_DAYS = 365            # crypto trades every day of the year
VAR_CONFIDENCE = (0.95, 0.99)
ROLLING_VOL_WINDOW = 30
CORRELATION_WINDOWS = (30, 90)
MA_SHORT, MA_LONG = 50, 200   # preferred market-cycle moving averages
RSI_WINDOW = 14
FORECAST_HORIZON = 30         # days ahead for the ARIMA forecast
FORECAST_TEST_FRACTION = 0.15
STRESS_QUANTILE = 0.10        # worst 10% of benchmark days = "stressed"

# Correlation needs every column populated on the same day, so one dead feed can
# empty the whole sample. A column carrying less than this fraction of the best-
# covered column's observations is dropped instead of being allowed to do that.
MIN_CORRELATION_COVERAGE = 0.5


def ma_windows(n_obs: int) -> tuple[int, int]:
    """
    Choose the moving-average windows that the loaded history can actually
    support.

    A 200-day MA needs 200 observations before it produces a single value, so on
    a 90- or 180-day window the classic 50/200 pair yields nothing at all and
    every day is labelled "Undetermined". The rule below keeps the canonical
    50/200 whenever there is comfortably enough history (>= 300 days) and
    otherwise scales both windows down proportionally, preserving the 1:4 ratio
    between them.

    The windows actually in use are surfaced in the dashboard so the label is
    never quietly computed on a different basis than the reader assumes.
    """
    if n_obs >= MA_LONG * 1.5:
        return MA_SHORT, MA_LONG
    long = max(20, n_obs // 3)
    short = max(5, long // 4)
    return short, long


# --------------------------------------------------------------------------- #
# Visual identity
#
# Five colours carry the entire interface:
#
#   INK          #212121   charcoal
#   ACCENT       #00BCD4   cyan — the focus colour
#   ACCENT_DARK  #0097A7   cyan, pressed
#   MUTED        #607D8B   blue-grey
#   MUTED_LIGHT  #90A4AE   blue-grey, light
#
# THEME decides which of them is background and which is foreground. In dark
# mode charcoal becomes the page and the two pale tints become text, so they act
# as ink rather than as fill — a few thousand thin glyphs, not a wash of colour.
# Switch to "light" and every chart and every panel follows; also swap the two
# marked lines in .streamlit/config.toml so Streamlit's own widgets match.
# --------------------------------------------------------------------------- #
THEME = "dark"          # "dark" or "light"

INK = "#212121"
ACCENT = "#00BCD4"
ACCENT_DARK = "#0097A7"
MUTED = "#607D8B"
MUTED_LIGHT = "#90A4AE"

_PALE = "#ECEFF1"       # ink in dark mode, surface in light mode
_PALE_DIM = "#CFD8DC"

_DARK = THEME == "dark"

# Semantic tokens. Everything downstream reads these, never the raw five.
PAGE_BG = INK if _DARK else "#FFFFFF"
PANEL_BG = INK if _DARK else _PALE
TEXT = _PALE if _DARK else INK
TEXT_MUTED = MUTED_LIGHT if _DARK else MUTED
BORDER = MUTED if _DARK else _PALE_DIM
GRID = "rgba(96,125,139,0.35)" if _DARK else _PALE_DIM
SERIES_PRIMARY = _PALE if _DARK else INK        # price lines, actuals
SERIES_TERTIARY = MUTED if _DARK else MUTED_LIGHT
SURFACE = PANEL_BG

# Translucent fills, derived from the five colours so no new hue enters.
FILL_ACCENT = "rgba(0, 188, 212, 0.18)" if _DARK else "rgba(0, 188, 212, 0.12)"
FILL_MUTED = "rgba(96, 125, 139, 0.28)" if _DARK else "rgba(96, 125, 139, 0.20)"

# Five cycle phases, five colours. The bear phase takes whichever end of the
# scheme contrasts with the page, so it stays legible in both themes.
PHASE_COLORS: dict[str, str] = {
    "Markup (Bull)": ACCENT,
    "Accumulation": ACCENT_DARK,
    "Distribution": MUTED_LIGHT,
    "Transition": MUTED,
    "Markdown (Bear)": _PALE if _DARK else INK,
}
UNDETERMINED_COLOR = BORDER   # bar chart only, never the ribbon

# Diverging, and the midpoint is the page colour itself: a correlation of zero
# fades into the background while strong relationships in either direction come
# forward. Blue-grey for negative, cyan for positive.
CORRELATION_SCALE = [
    [0.0, MUTED_LIGHT if _DARK else MUTED],
    [0.5, PAGE_BG],
    [1.0, ACCENT],
]
CELL_TEXT = TEXT
