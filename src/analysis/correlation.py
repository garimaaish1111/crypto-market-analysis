"""
Correlation analysis between digital and traditional assets.

Answers the question a portfolio holder actually cares about: is crypto
diversifying this portfolio, or is it just adding volatility that shows up at
the worst possible moment?

Two details in here carry most of the methodological weight.

*Alignment.* Crypto trades 365 days a year; equities, gold and oil trade roughly
252. The frames are joined on price with an inner join and differenced
afterwards, so a Friday-to-Monday move spans the same three calendar days for
both sides. Forward-filling macro prices across the weekend instead would
manufacture artificial zero-return days for the traditional assets and drag the
measured correlation toward zero.

*Yields are not prices.* A 10-year Treasury yield moving from 4.00% to 4.10% is a
10 basis-point change, not a "+2.5% return". Yield columns are differenced in
percentage points; everything else is percent-changed. Correlation is
scale-invariant, so the two mix without further adjustment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config


def _to_returns(prices: pd.DataFrame, yield_columns: set[str] | None = None) -> pd.DataFrame:
    """Percent-change every column except rate columns, which are differenced."""
    yield_columns = yield_columns or config.YIELD_ASSETS
    out = {}
    for column in prices.columns:
        series = prices[column]
        out[column] = series.diff() if column in yield_columns else series.pct_change()
    return pd.DataFrame(out, index=prices.index).dropna(how="all")


def drop_thin_columns(
    returns: pd.DataFrame, min_coverage: float = config.MIN_CORRELATION_COVERAGE
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove columns too sparsely populated to correlate, before rows are dropped.

    Returns ``(kept_frame, dropped_column_names)``.

    This guards the failure mode that matters in practice: a single provider
    outage. When yfinance rate-limits one ticker it still returns the frame, just
    with that column entirely NaN — and because correlation needs every column
    populated on the same day, the row-wise ``dropna`` below would then discard
    *every* row and leave an empty sample. One rate-limited commodity would take
    the whole correlation tab down with it.

    Dropping the unusable column first keeps the other ten assets correlating
    normally, which is the same fail-open principle the data loaders follow.
    Coverage is measured against the best-populated column rather than the row
    count, so a short history is not mistaken for a broken feed.
    """
    if returns.empty:
        return returns, []

    counts = returns.notna().sum()
    best = int(counts.max()) if len(counts) else 0
    if best == 0:
        return returns.iloc[:, :0], list(returns.columns)

    keep = [c for c in returns.columns if counts[c] >= best * min_coverage]
    dropped = [c for c in returns.columns if c not in keep]
    return returns[keep], dropped


def align_returns(crypto_prices: pd.DataFrame, macro_prices: pd.DataFrame) -> pd.DataFrame:
    """Join crypto and macro prices on common dates, then convert to returns."""
    combined = crypto_prices.join(macro_prices, how="inner")
    returns, _ = drop_thin_columns(_to_returns(combined))
    return returns.dropna()


def correlation_matrix(returns: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """
    Correlation matrix of the aligned return columns.

    ``pearson`` measures linear co-movement; ``spearman`` ranks first and is far
    less distorted by the extreme single-day moves crypto produces routinely.
    Comparing the two is a cheap robustness check — a large gap between them
    means the Pearson figure is being driven by a handful of outliers.
    """
    return returns.corr(method=method)


def rolling_correlation(
    returns: pd.DataFrame, asset_a: str, asset_b: str, window: int = 30
) -> pd.Series:
    """
    Rolling correlation between two named columns.

    A single coefficient over the whole sample hides the structural break that
    matters: crypto-versus-equity correlation sat near zero for years and rose
    sharply once institutional capital arrived. Only the rolling view shows it.
    """
    return returns[asset_a].rolling(window).corr(returns[asset_b]).dropna()


def beta(returns: pd.DataFrame, asset: str, benchmark: str) -> float:
    """
    Beta of ``asset`` to ``benchmark`` — the OLS slope of one on the other.

    Beta above 1 means the asset amplifies the benchmark's moves, which is the
    signature of a high-beta risk asset rather than a hedge.
    """
    sub = returns[[asset, benchmark]].dropna()
    if len(sub) < 3 or sub[benchmark].var() == 0:
        return float("nan")
    covariance = np.cov(sub[asset], sub[benchmark])[0, 1]
    return float(covariance / sub[benchmark].var())


def crypto_macro_summary(
    returns: pd.DataFrame,
    crypto_cols: list[str],
    macro_cols: list[str],
    method: str = "pearson",
) -> pd.DataFrame:
    """Compact table: rows = crypto assets, columns = macro assets."""
    corr = correlation_matrix(returns, method=method)
    present_crypto = [c for c in crypto_cols if c in corr.columns]
    present_macro = [m for m in macro_cols if m in corr.columns]
    return corr.loc[present_crypto, present_macro].round(2)


@dataclass
class RegimeCorrelation:
    """Correlation measured separately in calm and stressed markets."""

    asset: str
    benchmark: str
    calm: float
    stressed: float
    calm_days: int
    stressed_days: int

    @property
    def shift(self) -> float:
        """How much the correlation moves when the benchmark is falling hard."""
        return self.stressed - self.calm

    def as_dict(self) -> dict[str, object]:
        def fmt(value: float) -> str:
            return "—" if pd.isna(value) else f"{value:+.2f}"

        return {
            "Asset": self.asset,
            f"Calm ({self.calm_days}d)": fmt(self.calm),
            f"Stressed ({self.stressed_days}d)": fmt(self.stressed),
            "Change": fmt(self.shift),
        }


def regime_correlation(
    returns: pd.DataFrame,
    asset: str,
    benchmark: str = config.STRESS_BENCHMARK,
    quantile: float = config.STRESS_QUANTILE,
) -> RegimeCorrelation:
    """
    Split the sample on the benchmark's worst days and correlate within each half.

    Diversification is only worth anything if it survives a crash, so the honest
    test is not the full-sample correlation but whether the relationship holds
    when the benchmark is falling. Correlations across asset classes have a
    well-documented tendency to converge toward 1 in exactly those periods —
    March 2020, when Bitcoin fell roughly 50% alongside global equities, is the
    cleanest recent illustration.
    """
    if asset not in returns.columns or benchmark not in returns.columns:
        return RegimeCorrelation(asset, benchmark, float("nan"), float("nan"), 0, 0)

    sub = returns[[asset, benchmark]].dropna()
    if len(sub) < 20:
        return RegimeCorrelation(asset, benchmark, float("nan"), float("nan"), 0, 0)

    threshold = sub[benchmark].quantile(quantile)
    stressed_mask = sub[benchmark] <= threshold

    stressed = sub[stressed_mask]
    calm = sub[~stressed_mask]

    def safe_corr(frame: pd.DataFrame) -> float:
        if len(frame) < 3:
            return float("nan")
        return float(frame[asset].corr(frame[benchmark]))

    return RegimeCorrelation(
        asset=asset,
        benchmark=benchmark,
        calm=safe_corr(calm),
        stressed=safe_corr(stressed),
        calm_days=len(calm),
        stressed_days=len(stressed),
    )


def regime_table(
    returns: pd.DataFrame,
    assets: list[str],
    benchmark: str = config.STRESS_BENCHMARK,
) -> pd.DataFrame:
    """Calm-versus-stressed correlation for every crypto asset in one table."""
    rows = [regime_correlation(returns, a, benchmark).as_dict() for a in assets if a in returns.columns]
    return pd.DataFrame(rows).set_index("Asset") if rows else pd.DataFrame()
