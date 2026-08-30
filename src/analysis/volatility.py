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

import warnings
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


# --------------------------------------------------------------------------- #
# GARCH(1,1)
#
# Rolling standard deviation is what the metrics above use, and it has two
# well-known weaknesses: every day inside the window carries equal weight, and a
# large move stays in the estimate at full strength until it falls out of the
# far end, then vanishes in one step. Realised crypto volatility does neither.
# It clusters — a violent day makes the next day more likely to be violent — and
# it decays smoothly.
#
# GARCH(1,1) models exactly that: tomorrow's variance is a weighted sum of a
# long-run level, yesterday's surprise, and yesterday's variance. Two parameters
# summarise the behaviour:
#
#   alpha  how sharply volatility reacts to a new shock
#   beta   how long that shock persists
#
# alpha + beta is the persistence. Close to 1 means shocks decay slowly, which is
# the normal finding for daily financial returns.
# --------------------------------------------------------------------------- #


@dataclass
class GarchResult:
    """A fitted GARCH(1,1) and the pieces of it worth putting on screen."""

    symbol: str
    omega: float
    alpha: float
    beta: float

    conditional_volatility: pd.Series   # annualised, aligned to the return index
    forecast: pd.Series                 # annualised, forward-looking
    converged: bool = True
    message: str = ""

    @property
    def persistence(self) -> float:
        """alpha + beta. At or above 1 the process has no finite long-run variance."""
        return self.alpha + self.beta

    @property
    def is_stationary(self) -> bool:
        return self.persistence < 1.0

    @property
    def long_run_volatility(self) -> float:
        """Annualised unconditional volatility the process reverts to."""
        if not self.is_stationary or self.omega <= 0:
            return float("nan")
        daily_variance = self.omega / (1 - self.persistence)
        return float(np.sqrt(daily_variance) * np.sqrt(config.TRADING_DAYS) / 100)

    @property
    def current_volatility(self) -> float:
        if self.conditional_volatility.empty:
            return float("nan")
        return float(self.conditional_volatility.iloc[-1])

    def verdict(self) -> str:
        """One line a non-technical reader can act on."""
        if not self.converged:
            return f"The GARCH fit did not converge: {self.message}"

        current, long_run = self.current_volatility, self.long_run_volatility
        if not np.isfinite(current):
            return "No conditional volatility estimate is available."

        if not self.is_stationary:
            return (
                f"Persistence is {self.persistence:.3f}, at or above 1, so shocks "
                "do not decay and no long-run level exists. Read the conditional "
                "series but not the mean reversion."
            )

        direction = "above" if current > long_run else "below"
        return (
            f"Volatility is currently {current:.0%}, {direction} its long-run level "
            f"of {long_run:.0%}. Persistence is {self.persistence:.3f}: a shock "
            f"today still carries {self.persistence ** 30:.0%} of its force in a month."
        )


def fit_garch(
    price: pd.Series, symbol: str = "", horizon: int = 30
) -> GarchResult:
    """
    Fit GARCH(1,1) to daily returns and project conditional volatility forward.

    Returns are scaled by 100 before fitting. This is not cosmetic: daily returns
    are order 0.01, their variance order 0.0001, and the optimiser converges
    poorly on parameters that small. Everything is scaled back on the way out.

    Never raises. If the fit fails the result carries ``converged=False`` and the
    reason, so the dashboard can fall back to the rolling estimate rather than
    losing the tab.
    """
    empty = pd.Series(dtype=float)

    try:
        from arch import arch_model
    except ImportError:
        return GarchResult(
            symbol, float("nan"), float("nan"), float("nan"), empty, empty,
            converged=False,
            message="the `arch` package is not installed (pip install arch)",
        )

    rets = daily_returns(price).dropna() * 100
    if len(rets) < 60:
        return GarchResult(
            symbol, float("nan"), float("nan"), float("nan"), empty, empty,
            converged=False,
            message=f"only {len(rets)} return observations; GARCH needs at least 60",
        )

    try:
        with warnings.catch_warnings():
            # Scoped, not global: statsmodels and arch both emit convergence
            # chatter that would otherwise silence warnings process-wide.
            warnings.simplefilter("ignore")
            model = arch_model(rets, vol="GARCH", p=1, q=1, mean="Constant", dist="t")
            fitted = model.fit(disp="off", show_warning=False)

            params = fitted.params
            omega = float(params.get("omega", float("nan")))
            alpha = float(params.get("alpha[1]", float("nan")))
            beta = float(params.get("beta[1]", float("nan")))

            # Back to fractions, then annualised.
            annualiser = np.sqrt(config.TRADING_DAYS) / 100
            conditional = fitted.conditional_volatility * annualiser
            conditional = pd.Series(np.asarray(conditional), index=rets.index)

            projection = fitted.forecast(horizon=horizon, reindex=False)
            variance = np.asarray(projection.variance.iloc[-1])
            future_index = pd.date_range(
                price.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D"
            )
            forward = pd.Series(np.sqrt(variance) * annualiser, index=future_index)

        return GarchResult(
            symbol=symbol,
            omega=omega,
            alpha=alpha,
            beta=beta,
            conditional_volatility=conditional,
            forecast=forward,
            converged=True,
        )
    except Exception as exc:  # noqa: BLE001 - a failed fit must not take the tab down
        return GarchResult(
            symbol, float("nan"), float("nan"), float("nan"), empty, empty,
            converged=False, message=str(exc),
        )
