"""
Market-cycle phase identification.

Crypto markets are commonly described in four Wyckoff-style phases:

    Accumulation -> Markup (bull) -> Distribution -> Markdown (bear)

The phase is inferred from three transparent signals — trend structure (price
against its short and long moving averages), momentum (RSI), and drawdown from
the running peak. The rules are deliberately explainable rather than a black
box, which matters for a tool an investor is meant to trust: every label can be
traced back to the specific condition that produced it.

Two things changed from the first version and both are worth knowing:

*Adaptive windows.* A 200-day moving average produces nothing at all on a 90- or
180-day history, so every day came back "Undetermined". The windows now scale
with the amount of history available (see ``config.ma_windows``) and the values
actually in use are reported alongside the label.

*Tightened rules.* Two rules previously fired in situations they were not meant
to describe: a soft-momentum uptrend was labelled Distribution even when price
was far from its highs, and any shallow downtrend was labelled Accumulation.
Distribution now additionally requires price to be near the peak, and a
downtrend only reads as Accumulation once a deep drawdown is already in place.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config

# Thresholds, named so the rule table below reads as prose.
RSI_STRONG = 55        # above this, momentum is confirming the advance
RSI_WEAK = 45          # below this, momentum is confirming the decline
NEAR_HIGHS = -0.10     # within 10% of the peak
DEEP_DRAWDOWN = -0.25  # a bear-market-scale decline
BASING_DRAWDOWN = -0.30


def moving_averages(
    price: pd.Series, short: int | None = None, long: int | None = None
) -> pd.DataFrame:
    """Return price plus its short and long simple moving averages."""
    if short is None or long is None:
        short, long = config.ma_windows(len(price))
    return pd.DataFrame(
        {
            "price": price,
            "ma_short": price.rolling(short).mean(),
            "ma_long": price.rolling(long).mean(),
        }
    )


def rsi(price: pd.Series, window: int = config.RSI_WINDOW) -> pd.Series:
    """
    Relative Strength Index using Wilder's smoothing.

    ``adjust=False`` is required for Wilder's recursive average; pandas defaults
    to ``adjust=True``, which is a different weighting scheme and yields slightly
    different values.

    The two degenerate cases are handled explicitly rather than being swept into
    a neutral fill: an unbroken run of gains (average loss of zero) is maximum
    strength at 100, and a completely flat stretch is neutral at 50. The previous
    version returned 50 for both, which quietly reported the strongest possible
    rally as neutral momentum.
    """
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)

    pure_gains = (avg_loss == 0) & (avg_gain > 0)
    flat = (avg_loss == 0) & (avg_gain == 0)
    out = out.mask(pure_gains, 100.0).mask(flat, 50.0)

    # Values inside the warm-up window stay NaN. Filling them with a neutral 50
    # would fabricate momentum readings for days that have none.
    return out


def phase_series(
    price: pd.Series, short: int | None = None, long: int | None = None
) -> pd.Series:
    """
    Label every day with its inferred cycle phase.

    Vectorised with ``np.select`` — the earlier row-by-row loop with ``.loc``
    lookups recomputed the same series several times per dashboard tab.
    """
    if short is None or long is None:
        short, long = config.ma_windows(len(price))

    ma = moving_averages(price, short, long)
    momentum = rsi(price)
    drawdown = price / price.cummax() - 1

    p, ma_s, ma_l = ma["price"], ma["ma_short"], ma["ma_long"]
    uptrend = (p > ma_s) & (ma_s > ma_l)
    downtrend = (p < ma_s) & (ma_s < ma_l)
    near_highs = drawdown > NEAR_HIGHS

    conditions = [
        # Trending up with momentum confirming: the advance itself.
        uptrend & (momentum >= RSI_STRONG),
        # Trending up but stalling near the highs: supply meeting demand.
        uptrend & (momentum < RSI_STRONG) & near_highs,
        # Trending down, weak, and already deeply below the peak: the decline.
        downtrend & (momentum <= RSI_WEAK) & (drawdown < DEEP_DRAWDOWN),
        # Trending down but momentum is recovering off a deep low: basing.
        downtrend & (momentum > RSI_WEAK) & (drawdown < DEEP_DRAWDOWN),
        # Any other downtrend is an early decline, not accumulation.
        downtrend,
        # Sideways, far off the highs, momentum subdued: quiet basing.
        (drawdown < BASING_DRAWDOWN) & (momentum < 50),
        # Sideways, near the highs, momentum firm: topping.
        near_highs & (momentum > RSI_STRONG),
    ]
    choices = [
        "Markup (Bull)",
        "Distribution",
        "Markdown (Bear)",
        "Accumulation",
        "Markdown (Bear)",
        "Accumulation",
        "Distribution",
    ]

    labels = np.select(conditions, choices, default="Transition")

    # Days without a long MA or without an RSI reading cannot be classified.
    undetermined = ma_l.isna().to_numpy() | momentum.isna().to_numpy()
    labels = np.where(undetermined, "Undetermined", labels)

    return pd.Series(labels, index=price.index, name="phase")


@dataclass
class CyclePhase:
    """Snapshot of an asset's current cycle position."""

    symbol: str
    phase: str
    rsi: float
    price_vs_ma_long: float      # fraction above/below the long MA
    drawdown_from_high: float    # against the window high, not an all-time high
    trend: str
    ma_short: int
    ma_long: int

    def as_dict(self) -> dict[str, object]:
        def fmt(value: float, spec: str) -> str:
            return "—" if pd.isna(value) else format(value, spec)

        return {
            "Symbol": self.symbol,
            "Cycle Phase": self.phase,
            f"RSI({config.RSI_WINDOW})": fmt(self.rsi, ".0f"),
            f"Price vs {self.ma_long}MA": fmt(self.price_vs_ma_long, "+.1%"),
            "Drawdown vs window high": fmt(self.drawdown_from_high, ".1%"),
            "Trend": self.trend,
        }


def current_phase(symbol: str, price: pd.Series) -> CyclePhase:
    """Summarise the latest cycle phase for one asset."""
    short, long = config.ma_windows(len(price))
    ma = moving_averages(price, short, long)
    latest = ma.iloc[-1]

    momentum = float(rsi(price).iloc[-1])
    ma_long_value = latest["ma_long"]
    price_vs_ma = (
        float(latest["price"] / ma_long_value - 1) if pd.notna(ma_long_value) else float("nan")
    )
    drawdown = float(price.iloc[-1] / price.cummax().iloc[-1] - 1)

    if pd.isna(ma_long_value):
        trend = "n/a"
    else:
        trend = "Up" if latest["price"] > ma_long_value else "Down"

    return CyclePhase(
        symbol=symbol,
        phase=str(phase_series(price, short, long).iloc[-1]),
        rsi=momentum,
        price_vs_ma_long=price_vs_ma,
        drawdown_from_high=drawdown,
        trend=trend,
        ma_short=short,
        ma_long=long,
    )
