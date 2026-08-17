"""
Volatility and risk analytics.

Given a daily price series this module computes the standard toolkit an investor
uses to size positions and assess downside: annualised volatility, rolling
volatility, Value-at-Risk / Conditional VaR, maximum drawdown, and the
Sharpe / Sortino ratios. All functions are pure and operate on pandas objects,
so they are reused unchanged by the dashboard and the notebook.

Annualisation uses 365 days, not the 252 used for equities: crypto trades every
day of the year.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config


def daily_returns(price: pd.Series) -> pd.Series:
    """Simple daily percentage returns."""
    return price.pct_change().dropna()


def log_returns(price: pd.Series) -> pd.Series:
    """Log returns (used for the time-series models)."""
    return np.log(price / price.shift(1)).dropna()


def annualised_volatility(price: pd.Series, trading_days: int = config.TRADING_DAYS) -> float:
    """Annualised standard deviation of daily returns."""
    return float(daily_returns(price).std() * np.sqrt(trading_days))


def rolling_volatility(
    price: pd.Series,
    window: int = config.ROLLING_VOL_WINDOW,
    trading_days: int = config.TRADING_DAYS,
) -> pd.Series:
    """Annualised rolling volatility series."""
    return daily_returns(price).rolling(window).std() * np.sqrt(trading_days)


def historical_var(price: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical one-day Value-at-Risk at ``confidence``.

    Returned as a positive number = the daily loss not expected to be exceeded
    with the given probability (0.05 means a 5% loss). Estimated
    non-parametrically from the empirical return distribution, which avoids the
    normal-distribution assumption that badly understates crypto's fat tails.
    """
    rets = daily_returns(price)
    if rets.empty:
        return float("nan")
    return float(-np.percentile(rets, (1 - confidence) * 100))


def conditional_var(price: pd.Series, confidence: float = 0.95) -> float:
    """Expected loss *given* the VaR threshold is breached (CVaR / expected shortfall)."""
    rets = daily_returns(price)
    if rets.empty:
        return float("nan")
    threshold = np.percentile(rets, (1 - confidence) * 100)
    tail = rets[rets <= threshold]
    return float(-tail.mean()) if len(tail) else float("nan")


def max_drawdown(price: pd.Series) -> float:
    """
    Largest peak-to-trough decline *within the loaded window* (negative fraction).

    Note this is a window high, not an all-time high: with a 365-day history the
    running maximum is the one-year peak. The dashboard labels it accordingly.
    """
    running_max = price.cummax()
    return float((price / running_max - 1).min())


def drawdown_series(price: pd.Series) -> pd.Series:
    """Full drawdown-from-peak series, for the underwater chart."""
    return price / price.cummax() - 1


def sharpe_ratio(
    price: pd.Series,
    risk_free: float = config.RISK_FREE_RATE,
    trading_days: int = config.TRADING_DAYS,
) -> float:
    """Annualised Sharpe ratio: excess return per unit of total volatility."""
    rets = daily_returns(price)
    if rets.empty:
        return float("nan")
    excess = rets - risk_free / trading_days
    denom = rets.std()
    return float(excess.mean() / denom * np.sqrt(trading_days)) if denom else float("nan")


def downside_deviation(
    returns: pd.Series,
    target: float = 0.0,
) -> float:
    """
    Downside deviation about ``target``: sqrt(mean(min(r - target, 0)^2)).

    Measured across *all* observations, not just the losing ones. Taking the
    standard deviation of the negative returns alone — a common shortcut — is a
    different quantity: it measures spread about the mean of the losses rather
    than shortfall against the target, and inflates the resulting Sortino ratio.
    """
    if returns.empty:
        return float("nan")
    shortfall = np.minimum(returns - target, 0.0)
    return float(np.sqrt(np.mean(shortfall**2)))


def sortino_ratio(
    price: pd.Series,
    risk_free: float = config.RISK_FREE_RATE,
    trading_days: int = config.TRADING_DAYS,
) -> float:
    """Annualised Sortino ratio — penalises only shortfall against the risk-free rate."""
    rets = daily_returns(price)
    if rets.empty:
        return float("nan")
    target = risk_free / trading_days
    excess = rets - target
    denom = downside_deviation(rets, target)
    return float(excess.mean() / denom * np.sqrt(trading_days)) if denom else float("nan")


def volatility_regime(price: pd.Series, window: int = config.ROLLING_VOL_WINDOW) -> str:
    """
    Classify the *current* rolling vol against its own history: Low / Elevated / High.

    The reference distribution excludes the current observation, so the reading is
    "where does today sit relative to what came before" rather than a percentile
    of a set that already contains today.
    """
    rv = rolling_volatility(price, window).dropna()
    if len(rv) < 5:
        return "Unknown"
    current = rv.iloc[-1]
    history = rv.iloc[:-1]
    low, high = history.quantile(0.33), history.quantile(0.66)
    if current <= low:
        return "Low"
    if current >= high:
        return "High"
    return "Elevated"


@dataclass
class RiskProfile:
    """Container for the headline risk metrics of a single asset."""

    symbol: str
    annual_volatility: float
    var_95: float
    var_99: float
    cvar_95: float
    max_drawdown: float
    sharpe: float
    sortino: float
    regime: str

    def as_dict(self) -> dict[str, object]:
        def fmt(value: float, spec: str) -> str:
            return "—" if pd.isna(value) else format(value, spec)

        return {
            "Symbol": self.symbol,
            "Annual Vol": fmt(self.annual_volatility, ".1%"),
            "VaR 95%": fmt(self.var_95, ".2%"),
            "VaR 99%": fmt(self.var_99, ".2%"),
            "CVaR 95%": fmt(self.cvar_95, ".2%"),
            "Max Drawdown": fmt(self.max_drawdown, ".1%"),
            "Sharpe": fmt(self.sharpe, ".2f"),
            "Sortino": fmt(self.sortino, ".2f"),
            "Vol Regime": self.regime,
        }


def risk_profile(symbol: str, price: pd.Series) -> RiskProfile:
    """Compute the full risk profile for one asset in one call."""
    return RiskProfile(
        symbol=symbol,
        annual_volatility=annualised_volatility(price),
        var_95=historical_var(price, 0.95),
        var_99=historical_var(price, 0.99),
        cvar_95=conditional_var(price, 0.95),
        max_drawdown=max_drawdown(price),
        sharpe=sharpe_ratio(price),
        sortino=sortino_ratio(price),
        regime=volatility_regime(price),
    )
