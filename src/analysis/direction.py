"""
Next-day direction classification.

This module exists to answer a question the rest of the project raises but does
not settle: the Fear & Greed index and the Bitcoin on-chain series are collected
on every run and then displayed and nothing more. If they carry information
about tomorrow, a model given them should beat a model without them, and should
beat the trivial rule of always predicting the majority class.

The evaluation follows the same discipline as ``forecast.py``:

*Time-ordered splits only.* ``TimeSeriesSplit`` trains on the past and tests on
the future, always. A random split on a time series lets the model learn from
days that had not happened yet, which is the single most common way a financial
ML result turns out to be fictional.

*A baseline that is hard to beat by accident.* Accuracy alone is meaningless
when one class is more common: an asset that rose on 54% of days makes "always
predict up" a 54% model. Every score here is reported against that majority-class
base rate, and the headline number is the lift over it, not the raw accuracy.

*Features are lagged.* Every predictor is computed from information available at
the close of day t, and the target is the sign of the return from t to t+1.
Nothing in the feature matrix is derived from the day being predicted.

The expected outcome is a small lift or none at all. Daily crypto direction is
close to unpredictable, and a model that says so honestly is a more useful
deliverable than one that reports 95% accuracy because it leaked.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
from src.analysis import cycles, volatility

# Feature groups, named so the dashboard can report which block each came from
# and so the "do the unused feeds help?" comparison has something to switch off.
TECHNICAL_FEATURES = [
    "rsi",
    "vol_30d",
    "drawdown",
    "price_vs_ma",
    "return_1d",
    "return_7d",
]
SENTIMENT_FEATURES = ["fng_level", "fng_change_7d"]
ONCHAIN_FEATURES = ["tx_count_z", "active_addresses_z"]
MACRO_FEATURES = ["sp500_return", "dxy_change", "yield_change"]

ALL_FEATURES = TECHNICAL_FEATURES + SENTIMENT_FEATURES + ONCHAIN_FEATURES + MACRO_FEATURES


def _zscore(series: pd.Series, window: int = 30) -> pd.Series:
    """Rolling z-score. Absolute on-chain counts trend; their surprise does not."""
    rolling = series.rolling(window)
    return (series - rolling.mean()) / rolling.std()


def build_features(
    price: pd.Series,
    sentiment: pd.DataFrame | None = None,
    onchain: pd.DataFrame | None = None,
    macro_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Assemble the predictor matrix from everything the pipeline already loads.

    Every column is computed from data available at the close of the row's own
    day. The caller pairs this with ``build_target``, which looks one day ahead.
    Blocks whose source feed is missing are simply absent from the result, so a
    dead feed costs those columns rather than the whole model.
    """
    short, _long = config.ma_windows(len(price))
    features = pd.DataFrame(index=price.index)

    # --- Technical: derived from price alone ------------------------------- #
    features["rsi"] = cycles.rsi(price)
    features["vol_30d"] = volatility.rolling_volatility(price, window=30)
    features["drawdown"] = price / price.cummax() - 1
    # The SHORT average, deliberately. Using the long one would cost 199 rows of
    # warm-up on a 365-day history and leave too little to cross-validate on.
    features["price_vs_ma"] = price / price.rolling(short).mean() - 1
    features["return_1d"] = price.pct_change()
    features["return_7d"] = price.pct_change(7)

    # --- Sentiment: the Fear & Greed feed, previously displayed only -------- #
    if sentiment is not None and "fng_value" in sentiment:
        fng = pd.to_numeric(sentiment["fng_value"], errors="coerce")
        fng = fng.reindex(price.index).ffill()
        features["fng_level"] = fng
        features["fng_change_7d"] = fng.diff(7)

    # --- On-chain: network usage, previously displayed only ----------------- #
    if onchain is not None and not onchain.empty:
        oc = onchain.reindex(price.index).ffill()
        if "tx_count" in oc:
            features["tx_count_z"] = _zscore(oc["tx_count"])
        if "active_addresses" in oc:
            features["active_addresses_z"] = _zscore(oc["active_addresses"])

    # --- Macro: risk-on context -------------------------------------------- #
    if macro_prices is not None and not macro_prices.empty:
        macro = macro_prices.reindex(price.index).ffill()
        if "S&P 500" in macro:
            features["sp500_return"] = macro["S&P 500"].pct_change()
        if "US Dollar Index" in macro:
            features["dxy_change"] = macro["US Dollar Index"].pct_change()
        if "US 10Y Yield" in macro:
            # A rate, not a price: differenced in percentage points.
            features["yield_change"] = macro["US 10Y Yield"].diff()

    return features.replace([np.inf, -np.inf], np.nan)


def build_target(price: pd.Series) -> pd.Series:
    """
    1 if tomorrow closes above today, 0 if not, NaN where tomorrow is unknown.

    Shifting by -1 is what makes this a forecast rather than a description: row
    t carries the outcome of day t+1, so the model is never shown the day it is
    being asked about.

    The final row has no tomorrow, and the naive expression for this label is
    quietly wrong there. ``price.shift(-1)`` is NaN on the last row; ``NaN > x``
    evaluates to False rather than propagating, and ``.astype(int)`` then turns
    that False into a confident 0 — a "closed down" label for a day whose
    outcome does not exist. Because the column is then free of NaN, no
    downstream ``dropna`` removes it, and the fabricated label survives into a
    scored test fold.

    Comparing first and masking afterwards keeps the unknown day unknown, so the
    caller's ``dropna`` drops it.
    """
    tomorrow = price.shift(-1)
    direction = (tomorrow > price).astype(float)
    return direction.mask(tomorrow.isna()).rename("direction")


@dataclass
class DirectionResult:
    """Cross-validated performance of one feature set, against the base rate."""

    symbol: str
    model_name: str
    feature_names: list[str]

    accuracy: float
    base_rate: float
    roc_auc: float
    precision: float
    recall: float

    fold_accuracies: list[float] = field(default_factory=list)
    importances: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    confusion: np.ndarray | None = None
    n_train: int = 0
    n_test: int = 0

    @property
    def lift(self) -> float:
        """
        Percentage points of accuracy above always predicting the majority class.

        This, not accuracy, is the number worth reading. An asset that rose on
        54% of days makes 54% the score to beat, not 50%.
        """
        return self.accuracy - self.base_rate

    @property
    def beats_baseline(self) -> bool:
        return self.lift > 0

    def verdict(self) -> str:
        """One line a non-technical reader can act on."""
        if not np.isfinite(self.accuracy):
            return "Not enough data to evaluate a direction model."
        if self.lift > 0.03:
            return (
                f"{self.model_name} predicted next-day direction correctly on "
                f"{self.accuracy:.1%} of held-out days, {self.lift:+.1%} above the "
                f"{self.base_rate:.1%} base rate. Treat a lift this size with "
                "caution until it survives a longer sample."
            )
        if self.lift > 0:
            return (
                f"{self.model_name} scored {self.accuracy:.1%} against a "
                f"{self.base_rate:.1%} base rate — a lift of {self.lift:+.1%}, "
                "which is inside the noise for a sample this size."
            )
        return (
            f"{self.model_name} scored {self.accuracy:.1%} and did not beat the "
            f"{self.base_rate:.1%} base rate of always predicting the majority "
            "class. Daily direction in this asset is not predictable from these "
            "features — which is the expected result, not a bug."
        )


def _make_model(name: str):
    """Two models: one linear and interpretable, one able to find interactions."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "Logistic regression":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, C=0.1, random_state=0)),
            ]
        )
    if name == "Gradient boosting":
        return GradientBoostingClassifier(
            n_estimators=120, max_depth=2, learning_rate=0.05, random_state=0
        )
    raise ValueError(f"unknown model: {name}")


def _importances(model, feature_names: list[str]) -> pd.Series:
    """
    Coefficients for the linear model, impurity gain for the ensemble.

    The final estimator is pulled out by checking for Pipeline explicitly rather
    than for __getitem__: GradientBoostingClassifier also defines __getitem__,
    and indexing into it returns a single tree rather than the fitted ensemble.
    """
    from sklearn.pipeline import Pipeline

    estimator = model[-1] if isinstance(model, Pipeline) else model
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_.ravel())
    else:
        return pd.Series(dtype=float)
    return pd.Series(values, index=feature_names).sort_values(ascending=False)


def evaluate_direction_model(
    features: pd.DataFrame,
    target: pd.Series,
    symbol: str = "BTC",
    model_name: str = "Gradient boosting",
    columns: list[str] | None = None,
    n_splits: int = 5,
) -> DirectionResult:
    """
    Walk-forward cross-validation of a direction classifier.

    ``TimeSeriesSplit`` produces expanding windows that always train on days
    earlier than the ones they are scored on. Predictions from every fold are
    pooled before scoring so the reported accuracy covers one continuous
    out-of-sample stretch rather than an average of small, noisy fold scores.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import TimeSeriesSplit

    columns = [c for c in (columns or list(features.columns)) if c in features.columns]
    frame = features[columns].join(target).dropna()

    if len(frame) < 60 or frame["direction"].nunique() < 2:
        return DirectionResult(
            symbol=symbol, model_name=model_name, feature_names=columns,
            accuracy=float("nan"), base_rate=float("nan"), roc_auc=float("nan"),
            precision=float("nan"), recall=float("nan"),
        )

    X = frame[columns].to_numpy()
    y = frame["direction"].to_numpy()

    splitter = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(frame) // 40)))
    pooled_true: list[int] = []
    pooled_pred: list[int] = []
    pooled_proba: list[float] = []
    fold_accuracies: list[float] = []
    fitted = None
    n_train = 0

    for train_idx, test_idx in splitter.split(X):
        model = _make_model(model_name)
        model.fit(X[train_idx], y[train_idx])
        predicted = model.predict(X[test_idx])

        pooled_true.extend(y[test_idx].tolist())
        pooled_pred.extend(predicted.tolist())
        if hasattr(model, "predict_proba"):
            pooled_proba.extend(model.predict_proba(X[test_idx])[:, 1].tolist())
        fold_accuracies.append(float(accuracy_score(y[test_idx], predicted)))
        fitted, n_train = model, len(train_idx)

    truth = np.asarray(pooled_true)
    predictions = np.asarray(pooled_pred)

    # The base rate is measured on the SAME held-out days the model was scored
    # on, not on the full history — otherwise the comparison is not like for like.
    majority = 1 if truth.mean() >= 0.5 else 0
    base_rate = float((truth == majority).mean())

    try:
        auc = float(roc_auc_score(truth, pooled_proba)) if pooled_proba else float("nan")
    except ValueError:
        auc = float("nan")

    return DirectionResult(
        symbol=symbol,
        model_name=model_name,
        feature_names=columns,
        accuracy=float(accuracy_score(truth, predictions)),
        base_rate=base_rate,
        roc_auc=auc,
        precision=float(precision_score(truth, predictions, zero_division=0)),
        recall=float(recall_score(truth, predictions, zero_division=0)),
        fold_accuracies=fold_accuracies,
        importances=_importances(fitted, columns) if fitted is not None else pd.Series(dtype=float),
        confusion=confusion_matrix(truth, predictions),
        n_train=n_train,
        n_test=len(truth),
    )


def feature_block_ablation(
    features: pd.DataFrame,
    target: pd.Series,
    symbol: str = "BTC",
    model_name: str = "Gradient boosting",
) -> pd.DataFrame:
    """
    Does adding the previously-unused feeds actually help?

    Trains the same model on technical features alone, then on technical plus
    each additional block, and reports accuracy against the base rate for each.
    This is the direct test of whether collecting Fear & Greed and on-chain data
    was worth doing, and it is designed to be able to answer "no".
    """
    blocks = {
        "Technical only": TECHNICAL_FEATURES,
        "+ Sentiment": TECHNICAL_FEATURES + SENTIMENT_FEATURES,
        "+ On-chain": TECHNICAL_FEATURES + ONCHAIN_FEATURES,
        "+ Macro": TECHNICAL_FEATURES + MACRO_FEATURES,
        "All features": ALL_FEATURES,
    }

    rows = []
    for label, columns in blocks.items():
        available = [c for c in columns if c in features.columns]
        if not available:
            continue
        result = evaluate_direction_model(
            features, target, symbol=symbol, model_name=model_name, columns=available
        )
        rows.append(
            {
                "Feature set": label,
                "Features": len(available),
                "Accuracy": result.accuracy,
                "Base rate": result.base_rate,
                "Lift": result.lift,
                "ROC AUC": result.roc_auc,
            }
        )
    return pd.DataFrame(rows).set_index("Feature set")
