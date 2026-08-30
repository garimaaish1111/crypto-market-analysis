"""
Data preprocessing and quality reporting.

Every feed in this project arrives from a different provider with a different
idea of what a clean daily series looks like. CoinGecko returns intraday points
that need collapsing to one row per day. Yahoo Finance skips weekends and
holidays, and rate-limits individual tickers into all-NaN columns. The
Blockchain.info charts API occasionally repeats a timestamp. alternative.me
returns its values as *strings*.

Rather than scattering ad-hoc ``.dropna()`` and ``.astype(float)`` calls through
the loaders, cleaning happens here, in one place, and every step reports what it
did. The report matters as much as the cleaning: silently discarding a third of
a dataset is how an analysis ends up describing something other than the market.

The functions are pure -- frame in, cleaned frame plus report out -- so they are
testable without a network call, and the dashboard can show the reader exactly
what was done to the data before any statistic was computed from it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# A daily return beyond this many standard deviations is flagged for review.
# Crypto genuinely moves 10-15% in a day, so this is deliberately loose: the aim
# is to catch provider glitches (a decimal point in the wrong place, a zero
# price) rather than to sand the fat tails off a real distribution.
OUTLIER_SIGMA = 8.0

# Below this fraction of the expected daily observations, a feed is reported as
# having a coverage problem rather than quietly being used.
MIN_COVERAGE = 0.5


@dataclass
class CleaningReport:
    """A record of what preprocessing did to one dataset."""

    dataset: str
    rows_in: int = 0
    rows_out: int = 0

    duplicates_removed: int = 0
    non_numeric_coerced: int = 0
    missing_filled: int = 0
    missing_remaining: int = 0
    non_positive_removed: int = 0
    outliers_flagged: int = 0
    columns_dropped: list[str] = field(default_factory=list)

    date_start: pd.Timestamp | None = None
    date_end: pd.Timestamp | None = None
    expected_days: int = 0
    n_columns: int = 0

    @property
    def rows_removed(self) -> int:
        return max(0, self.rows_in - self.rows_out)

    @property
    def coverage(self) -> float:
        """Observed rows as a fraction of the days the window should contain."""
        if not self.expected_days:
            return float("nan")
        return self.rows_out / self.expected_days

    # A multi-ticker frame legitimately carries some missing values: the five
    # macro assets trade on different calendars and their histories start on
    # different dates, so the earliest rows are ragged. Only a substantial
    # fraction of holes indicates an actual problem.
    MAX_MISSING_FRACTION = 0.05

    @property
    def missing_fraction(self) -> float:
        """Remaining NaNs as a fraction of all cells."""
        cells = self.rows_out * max(1, self.n_columns)
        return self.missing_remaining / cells if cells else 0.0

    @property
    def is_healthy(self) -> bool:
        """
        Whether this dataset is fit to compute on.

        Business-day feeds legitimately cover only ~70% of calendar days, so the
        coverage threshold sits below that.
        """
        if self.rows_out == 0:
            return False
        if self.missing_fraction > self.MAX_MISSING_FRACTION:
            return False
        return not (np.isfinite(self.coverage) and self.coverage < MIN_COVERAGE)

    def summary(self) -> str:
        """One line for the dashboard."""
        if self.rows_out == 0:
            return f"{self.dataset}: no usable rows."
        parts = [f"{self.rows_out:,} rows"]
        if self.date_start is not None and self.date_end is not None:
            parts.append(f"{self.date_start:%d %b %Y} to {self.date_end:%d %b %Y}")
        if self.rows_removed:
            parts.append(f"{self.rows_removed} removed")
        if self.missing_filled:
            parts.append(f"{self.missing_filled} gaps filled")
        if self.outliers_flagged:
            parts.append(f"{self.outliers_flagged} outliers flagged")
        return f"{self.dataset}: " + ", ".join(parts) + "."

    def as_dict(self) -> dict[str, object]:
        """Row form, for the quality table on the Data tab."""
        return {
            "Dataset": self.dataset,
            "Rows": f"{self.rows_out:,}",
            "Coverage": "—" if not np.isfinite(self.coverage) else f"{self.coverage:.0%}",
            "Duplicates": self.duplicates_removed,
            "Coerced": self.non_numeric_coerced,
            "Gaps filled": self.missing_filled,
            "Still missing": self.missing_remaining,
            "Outliers": self.outliers_flagged,
            "Status": "OK" if self.is_healthy else "Check",
        }


def _coerce_numeric(frame: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """
    Convert any column that is numeric-in-disguise, and leave real text alone.

    alternative.me returns the Fear & Greed value as a string. Comparing strings
    numerically does not raise -- it silently produces nonsense -- so this is
    checked rather than assumed.

    Dtype is not a reliable way to spot the candidates. Under pandas 2 a string
    column arrives as ``object``; under pandas 3 it arrives as the dedicated
    ``str`` dtype, so a check for ``object`` silently skips exactly the column
    this function exists to fix. Instead every non-numeric column is *attempted*
    and the result kept only if it actually parsed, which behaves the same on
    both major versions. A genuinely categorical column such as ``fng_label``
    coerces to all-NaN and is therefore preserved untouched.
    """
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_numeric_dtype(out[column]):
            continue
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            continue

        converted = pd.to_numeric(out[column], errors="coerce")
        parsed = int(converted.notna().sum())
        if parsed == 0:
            continue  # real text, e.g. the Fear & Greed classification label

        report.non_numeric_coerced += parsed
        out[column] = converted
    return out


def clean_timeseries(
    frame: pd.DataFrame,
    dataset: str,
    expected_days: int = 0,
    resample_daily: bool = True,
    fill_gaps: bool = True,
    require_positive: bool = False,
    numeric_only: bool = True,
) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Clean one time-indexed dataset and report every step.

    The order is deliberate:

    1. **Sort and de-duplicate the index.** A repeated timestamp silently
       double-counts a day in every downstream statistic.
    2. **Coerce to numeric**, because a string column compares without raising.
    3. **Collapse to one row per day**, since CoinGecko returns intraday points.
    4. **Drop non-positive prices** where they are impossible, before any log or
       percentage change turns them into infinities.
    5. **Forward-fill short gaps.** Forward, never backward: back-filling would
       carry a later observation into an earlier day, which is look-ahead.
    6. **Flag outliers without deleting them.** An 8-sigma day in crypto is
       usually real, and quietly removing it would understate tail risk --
       precisely the thing the VaR and CVaR figures exist to measure.
    """
    report = CleaningReport(dataset=dataset, expected_days=expected_days)

    if frame is None or frame.empty:
        return pd.DataFrame(), report

    out = frame.copy()
    report.rows_in = len(out)

    # --- 1. index hygiene --------------------------------------------------- #
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[out.index.notna()]
    out = out.sort_index()

    duplicated = out.index.duplicated(keep="last")
    report.duplicates_removed = int(duplicated.sum())
    out = out[~duplicated]

    # --- 2. dtypes ---------------------------------------------------------- #
    if numeric_only:
        # Every column is offered to the coercer, which decides what is genuinely
        # numeric by trying it rather than by inspecting a dtype whose meaning
        # changed between pandas 2 and 3.
        out = _coerce_numeric(out, report)

    # --- 3. one row per day ------------------------------------------------- #
    if resample_daily and len(out) > 1:
        out = out.resample("D").last()

    # --- 4. impossible values ----------------------------------------------- #
    if require_positive:
        numeric = out.select_dtypes(include=[np.number])
        if not numeric.empty:
            bad = (numeric <= 0).any(axis=1)
            report.non_positive_removed = int(bad.sum())
            out = out[~bad]

    # --- 5. gaps ------------------------------------------------------------ #
    before = int(out.isna().sum().sum())
    if fill_gaps:
        # Forward only. bfill would move a later value into an earlier day.
        out = out.ffill()
    after = int(out.isna().sum().sum())
    report.missing_filled = max(0, before - after)

    # A leading NaN cannot be forward-filled, so drop rows that are still empty.
    out = out.dropna(how="all")
    report.missing_remaining = int(out.isna().sum().sum())

    # --- 6. outliers, flagged not removed ----------------------------------- #
    report.outliers_flagged = count_outliers(out)

    report.rows_out = len(out)
    report.n_columns = len(out.columns)
    if len(out):
        report.date_start = out.index.min()
        report.date_end = out.index.max()

    return out, report


def count_outliers(frame: pd.DataFrame, sigma: float = OUTLIER_SIGMA) -> int:
    """
    Count daily moves beyond ``sigma`` standard deviations, without removing any.

    Flagging rather than winsorising is a deliberate choice. Value-at-Risk and
    CVaR exist precisely to describe the tail of this distribution, so trimming
    the tail before measuring it would produce a comfortable number that is
    wrong in the one direction that matters.
    """
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty or len(numeric) < 30:
        return 0
    returns = numeric.pct_change().replace([np.inf, -np.inf], np.nan)
    std = returns.std()
    std = std[std > 0]
    if std.empty:
        return 0
    z = (returns[std.index] - returns[std.index].mean()).abs() / std
    return int((z > sigma).sum().sum())


def clean_crypto_frame(
    frame: pd.DataFrame, symbol: str, expected_days: int = 0
) -> tuple[pd.DataFrame, CleaningReport]:
    """Price/volume/market-cap for one coin. Prices must be strictly positive."""
    return clean_timeseries(
        frame,
        dataset=f"Crypto — {symbol}",
        expected_days=expected_days,
        resample_daily=True,
        fill_gaps=True,
        require_positive=True,
    )


def clean_macro_frame(
    frame: pd.DataFrame, expected_days: int = 0
) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Traditional assets.

    Not resampled to a daily grid: these markets are genuinely closed at
    weekends, and inventing rows for those days would manufacture zero-return
    observations that drag every measured correlation toward zero. Non-positive
    values are permitted because a yield can legitimately be zero or negative.
    """
    return clean_timeseries(
        frame,
        dataset="Macro — traditional assets",
        expected_days=expected_days,
        resample_daily=False,
        fill_gaps=True,
        require_positive=False,
    )


def clean_sentiment_frame(
    frame: pd.DataFrame, expected_days: int = 0
) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Fear & Greed.

    The value arrives as a string and the label is genuinely categorical, so the
    text column is preserved while the value is coerced to a number and clipped
    to the index's documented 0-100 range.
    """
    cleaned, report = clean_timeseries(
        frame,
        dataset="Sentiment — Fear & Greed",
        expected_days=expected_days,
        resample_daily=True,
        fill_gaps=True,
    )
    if "fng_value" in cleaned:
        out_of_range = int(
            ((cleaned["fng_value"] < 0) | (cleaned["fng_value"] > 100)).sum()
        )
        report.outliers_flagged += out_of_range
        cleaned["fng_value"] = cleaned["fng_value"].clip(0, 100)
    return cleaned, report


def clean_onchain_frame(
    frame: pd.DataFrame, expected_days: int = 0
) -> tuple[pd.DataFrame, CleaningReport]:
    """Bitcoin network activity. Counts cannot be negative or zero."""
    return clean_timeseries(
        frame,
        dataset="On-chain — Bitcoin network",
        expected_days=expected_days,
        resample_daily=True,
        fill_gaps=True,
        require_positive=True,
    )


def quality_table(reports: list[CleaningReport]) -> pd.DataFrame:
    """Every report as one table, for the Data & Sources tab."""
    if not reports:
        return pd.DataFrame()
    return pd.DataFrame([r.as_dict() for r in reports]).set_index("Dataset")
