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
# Currency
#
# The dashboard is denominated in Indian rupees. CoinGecko supports `inr` as a
# vs_currency natively, so crypto prices are *fetched* in rupees rather than
# converted — no rounding error and no stale cross-rate.
#
# Yahoo Finance quotes the S&P 500, gold and crude oil in US dollars, so those
# are converted with the daily USD/INR rate. Two macro series are deliberately
# left alone: the US Dollar Index is an index level, not a price in dollars,
# and the 10Y yield is a rate in percentage points. Multiplying either by an
# exchange rate would be meaningless.
#
# Set CURRENCY = "usd" to switch the whole interface back; every downstream
# format call and conversion reads these values.
# --------------------------------------------------------------------------- #
CURRENCY = os.getenv("CURRENCY", "inr")     # "inr" or "usd"
CURRENCY_SYMBOL = "₹" if CURRENCY == "inr" else "$"
CURRENCY_LABEL = CURRENCY.upper()

FX_TICKER = "USDINR=X"           # Yahoo Finance symbol for the conversion rate
USD_INR_FALLBACK = 95.4          # used only if the FX feed is unreachable

# Macro columns genuinely priced in US dollars, and so needing conversion.
USD_QUOTED_ASSETS: set[str] = {"S&P 500", "Gold", "Crude Oil"}

# --------------------------------------------------------------------------- #
# Data / API settings
# --------------------------------------------------------------------------- #
# Live CoinGecko fetching. "simulated" skips the network entirely and generates
# crypto series from sample_data.py instead; "live" fetches for real and falls
# back to generated data only if a request actually fails.
#
# This was previously set to "simulated" to avoid 429s from the keyless tier.
# The throttling, backoff and key support below make live fetching reliable, so
# live is now the default. Macro, sentiment and on-chain feeds are always live.
CRYPTO_MODE = os.getenv("CRYPTO_MODE", "live")   # "live" or "simulated"

# --------------------------------------------------------------------------- #
# CoinGecko credentials
#
# The key is optional. Resolution order:
#   1. COINGECKO_API_KEY environment variable
#   2. apikey.txt next to this file (one line, the key and nothing else)
#   3. blank -> keyless public API
#
# run.bat / run.sh already export the environment variable when apikey.txt is
# present, but reading the file here too means `streamlit run app.py` picks the
# key up as well, rather than silently falling back to keyless access.
# --------------------------------------------------------------------------- #
def _read_api_key() -> str:
    env = os.getenv("COINGECKO_API_KEY", "").strip()
    if env:
        return env
    key_file = ROOT_DIR / "apikey.txt"
    try:
        if key_file.exists():
            return key_file.read_text(encoding="utf-8").splitlines()[0].strip()
    except Exception:
        pass
    return ""


API_KEY = _read_api_key()
COINGECKO_PLAN = os.getenv("COINGECKO_PLAN", "demo")   # "demo" or "pro"

# Rate limits differ by an order of magnitude, so the spacing between calls
# should too. A demo key is a stable 100/min, so 0.8s between calls is ample
# headroom. Keyless access shares an IP-based pool documented as roughly
# 10-30 calls/minute; 2.5s spacing (24/min) sat inside that band on paper but
# not in practice — a measured six-coin cold start hit two 429s at 30s of
# backoff each (74.8s total, most of it spent recovering from a rate limit
# that better spacing would have avoided in the first place). 6.0s spacing
# (10/min) targets the *worst* documented case instead of the average one, so
# six coins land in roughly 30-36s with no 429s at all — slower per call, but
# faster overall because it never pays the 30s backoff tax. A 429 that still
# gets through is treated as a signal the shared pool is under real pressure
# right now, so the keyless backoff starts higher too, rather than retrying
# into the same wall immediately.
COINGECKO_MIN_INTERVAL = 0.8 if API_KEY else 6.0   # seconds between live calls
COINGECKO_MAX_ATTEMPTS = 4      # attempts before falling back to sample data
COINGECKO_BACKOFF = 3.0 if API_KEY else 5.0    # initial backoff, doubles each retry
COINGECKO_MAX_WAIT = 30.0       # cap on any single wait
FNG_URL = "https://api.alternative.me/fng/"               # crypto Fear & Greed
BLOCKCHAIN_CHARTS = "https://api.blockchain.info/charts"  # BTC on-chain metrics

DEFAULT_DAYS = 365            # history window (CoinGecko free tier max)
HISTORY_WINDOWS = (90, 180, 270, 365)
REQUEST_TIMEOUT = 20          # seconds
# How long a cached API response stays usable.
#
# Six hours is the right value while developing: responsive, and never stale
# within a working day. Use it with:
#
#   CACHE_TTL_HOURS=6 streamlit run app.py
#
# The shipped default is deliberately enormous, because this project is
# *submitted* as a folder and assessed at an unknown later date. `data/cache/`
# travels with it, pre-populated with real responses from every feed, and the
# reader will not have an API key. If the TTL expired before assessment they
# would be dropped onto the slow keyless path — or, on a machine with no
# network, onto generated data — through no fault of their own and with no
# indication that anything had changed.
#
# A snapshot does not go "off". It is dated, and the dashboard reports the date
# range of every feed on the Datasets tab, so what the reader sees is always
# clearly attributable to a moment in time. Ten years is simply "longer than
# this project will ever be looked at".
CACHE_TTL_HOURS = float(os.getenv("CACHE_TTL_HOURS", 24 * 365 * 10))   # 10 years


def coingecko_base() -> str:
    """Pro keys use a different host from demo and keyless access."""
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

# A correlation computed on a handful of days is not a measurement, but it
# renders identically to one computed on hundreds. On the 90-day window the
# stressed bucket holds about seven observations, and reporting that to two
# decimal places beside a 224-observation calm figure invites a comparison the
# data cannot support. Below this count the regime split reports nothing.
MIN_REGIME_OBSERVATIONS = 20

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
