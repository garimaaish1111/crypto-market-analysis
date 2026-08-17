"""
Deterministic synthetic-data generator used as an offline fallback.

The live fetchers (CoinGecko, Yahoo Finance, alternative.me, blockchain.info)
are the primary source of truth. When the network is unavailable or an API is
rate-limited, these functions produce plausible series so the dashboard is
always demonstrable.

Two properties matter here and both were wrong in the first version:

*Determinism.* Seeds are derived with ``zlib.crc32``, not the built-in
``hash()``. Python randomises string hashing per process, so a ``hash()``-derived
seed produced a different Bitcoin price history on every restart — the opposite
of the reproducibility this module claims.

*Correlation structure.* Assets are no longer independent random walks. Every
series is driven by a shared risk-on factor plus its own idiosyncratic noise, so
the correlation analysis shows a plausible structure offline instead of the flat
zero matrix that independent walks produce. This is simulated structure, not
measured structure, and the dashboard says so whenever a feed is on sample data.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

import config

# Rough starting level, annualised drift/vol, and loading on the shared crypto
# factor. Betas near 1 mean the coin moves almost entirely with the crypto
# market; lower betas leave more of the variance idiosyncratic.
# Volatilities are set near the levels these assets have actually realised —
# Bitcoin in the mid-forties, the smaller alts higher — rather than at the
# textbook "crypto is 65% volatile" figure, which over a full year produces
# ranges wide enough to look like a bug.
#
# ``latest`` is where each path *ends*, not where it starts. Anchoring the final
# value keeps the headline price on screen plausible: a random walk left to run
# for a year from a fixed start can drift to two or three times its opening
# level, which looks wrong on a dashboard however sound the mathematics.
#
# Rescaling by a constant does not alter a single return, so volatility, VaR,
# correlation, beta and every phase label are unchanged. Only the level moves.
_CRYPTO_PROFILE = {
    "BTC": dict(latest=95_000, drift=0.12, vol=0.45, beta=0.90),
    "ETH": dict(latest=3_400, drift=0.10, vol=0.55, beta=0.92),
    "SOL": dict(latest=180, drift=0.15, vol=0.80, beta=0.85),
    "BNB": dict(latest=620, drift=0.08, vol=0.48, beta=0.80),
    "XRP": dict(latest=2.20, drift=0.06, vol=0.68, beta=0.72),
    "ADA": dict(latest=0.85, drift=0.05, vol=0.72, beta=0.78),
}

# Loadings on the equity/risk factor. Gold is near-neutral, the dollar index is
# negative (a strong dollar is risk-off), oil is moderately pro-cyclical.
_MACRO_PROFILE = {
    "S&P 500": dict(latest=6_100, drift=0.10, vol=0.15, beta=0.90),
    "Gold": dict(latest=3_300, drift=0.08, vol=0.13, beta=0.05),
    "US Dollar Index": dict(latest=103, drift=0.0, vol=0.07, beta=-0.35),
    "US 10Y Yield": dict(latest=4.1, drift=0.0, vol=0.20, beta=0.25),
    "Crude Oil": dict(latest=78, drift=0.05, vol=0.35, beta=0.45),
}

# Correlation between the crypto factor and the equity factor. Roughly the level
# observed since institutional participation grew from 2020 onward.
_CRYPTO_EQUITY_RHO = 0.35

# Fixed base seed. Every stream below is derived from it deterministically.
_BASE_SEED = 20260808


def _seed_for(name: str, salt: int = 0) -> int:
    """Stable per-name seed. crc32 is deterministic across processes; hash() is not."""
    return (zlib.crc32(name.encode()) + _BASE_SEED + salt) % (2**32)


def _date_index(days: int) -> pd.DatetimeIndex:
    """Continuous daily index — crypto trades every day."""
    end = pd.Timestamp.today().normalize()
    return pd.date_range(end=end, periods=days, freq="D")


def _factors(days: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (equity_factor, crypto_factor) as standard normal shock series.

    The crypto factor is built as rho * equity + sqrt(1 - rho^2) * independent,
    which gives exactly ``_CRYPTO_EQUITY_RHO`` correlation while leaving both
    factors unit-variance.
    """
    rng = np.random.default_rng(_seed_for("factors"))
    equity = rng.standard_normal(days)
    independent = rng.standard_normal(days)
    rho = _CRYPTO_EQUITY_RHO
    crypto = rho * equity + np.sqrt(1 - rho**2) * independent
    return equity, crypto


def _factor_gbm(
    latest: float, drift: float, vol: float, beta: float, factor: np.ndarray, seed: int
) -> np.ndarray:
    """
    Geometric Brownian Motion whose shocks load on a shared factor, rescaled so
    the path ends at ``latest``.

    The shock is ``beta * factor + sqrt(1 - beta^2) * idiosyncratic``, which keeps
    unit variance regardless of beta, so ``vol`` still means what it says. The
    ``- 0.5 * vol**2`` term is the Ito correction: without it the exponential
    would drift faster than the stated ``drift``.

    The final rescaling multiplies every point by one constant, so it leaves all
    returns — and therefore every metric in this project — untouched.
    """
    days = len(factor)
    rng = np.random.default_rng(seed)
    idio = rng.standard_normal(days)
    residual_weight = np.sqrt(max(0.0, 1 - beta**2))
    z = beta * factor + residual_weight * idio

    dt = 1 / 365
    shocks = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z
    path = np.exp(np.cumsum(shocks))
    return latest * path / path[-1]


def crypto_ohlcv(symbol: str, days: int = config.DEFAULT_DAYS) -> pd.DataFrame:
    """Return a price/volume/market-cap frame for one crypto symbol."""
    profile = dict(_CRYPTO_PROFILE.get(symbol, dict(latest=100, drift=0.3, vol=0.8, beta=0.8)))
    idx = _date_index(days)
    _, crypto_factor = _factors(days)

    price = _factor_gbm(factor=crypto_factor, seed=_seed_for(symbol), **profile)

    rng = np.random.default_rng(_seed_for(symbol, salt=1))
    # Turnover rises on big moves, so volume scales with |return| over a baseline.
    rets = np.diff(price, prepend=price[0]) / price
    base_volume = profile["latest"] * 1e6
    volume = base_volume * (1 + 6 * np.abs(rets)) * rng.lognormal(0, 0.3, days)

    # A fixed circulating supply per asset — market cap should track price, not
    # wander independently as it did when the multiplier was redrawn each day.
    supply = rng.uniform(1.7e7, 2.0e7)
    market_cap = price * supply

    return pd.DataFrame(
        {"price": price, "volume": volume, "market_cap": market_cap}, index=idx
    )


def macro_prices(days: int = config.DEFAULT_DAYS) -> pd.DataFrame:
    """
    Return traditional-asset closing prices on a *business-day* index.

    Yahoo Finance only returns trading days, so the fallback must do the same.
    Generating on the full daily grid and then dropping weekends means a
    Friday-to-Monday move spans the same three calendar days for macro assets as
    it does for crypto, which is what the correlation alignment assumes.
    """
    idx = _date_index(days)
    equity_factor, _ = _factors(days)

    out = {}
    for name, profile in _MACRO_PROFILE.items():
        out[name] = _factor_gbm(factor=equity_factor, seed=_seed_for(name), **profile)

    frame = pd.DataFrame(out, index=idx)
    return frame[frame.index.dayofweek < 5]


def fear_greed(days: int = config.DEFAULT_DAYS) -> pd.DataFrame:
    """
    Return a synthetic crypto Fear & Greed index (0-100).

    Sentiment tracks the crypto factor with a lag, because the index is largely a
    reflection of recent price action rather than an independent signal.
    """
    idx = _date_index(days)
    rng = np.random.default_rng(_seed_for("fear_greed"))
    _, crypto_factor = _factors(days)

    momentum = pd.Series(crypto_factor).rolling(14, min_periods=1).mean().to_numpy()
    noise = rng.normal(0, 3, days)
    val = np.clip(50 + momentum * 40 + noise, 5, 95)

    labels = pd.cut(
        val,
        bins=[0, 25, 45, 55, 75, 100],
        labels=["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"],
    )
    return pd.DataFrame({"fng_value": val.round(0), "fng_label": labels}, index=idx)


def onchain_btc(days: int = config.DEFAULT_DAYS) -> pd.DataFrame:
    """Return synthetic Bitcoin on-chain metrics (tx count, active addresses)."""
    idx = _date_index(days)
    rng = np.random.default_rng(_seed_for("onchain"))

    tx = np.clip(300_000 + np.cumsum(rng.normal(0, 4_000, days)), 200_000, 500_000)
    addr = np.clip(800_000 + np.cumsum(rng.normal(0, 8_000, days)), 500_000, 1_200_000)

    return pd.DataFrame(
        {"tx_count": tx.round(0), "active_addresses": addr.round(0)}, index=idx
    )
