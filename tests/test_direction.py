"""
Direction-model tests.

The most important test in this file is ``test_target_looks_forward_not_backward``.
Every catastrophic result in financial machine learning traces back to the model
seeing the day it was asked to predict, and the cheapest place to catch that is a
unit test on the target construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.analysis import direction


@pytest.fixture(scope="module")
def parts(btc_price):
    """Features and target built from the deterministic generator."""
    from src.data import sample_data

    features = direction.build_features(
        btc_price,
        sentiment=sample_data.fear_greed(365),
        onchain=sample_data.onchain_btc(365),
        macro_prices=sample_data.macro_prices(365),
    )
    return features, direction.build_target(btc_price)


class TestTargetConstruction:
    def test_target_looks_forward_not_backward(self):
        """
        Row t must carry the outcome of day t+1. If this shifts the wrong way the
        model is handed the answer and every downstream metric becomes fiction.
        """
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        price = pd.Series([100.0, 110.0, 105.0, 105.0, 120.0], index=idx)
        target = direction.build_target(price)
        assert target.iloc[0] == 1   # 100 -> 110 rises
        assert target.iloc[1] == 0   # 110 -> 105 falls
        assert target.iloc[2] == 0   # 105 -> 105 is not a rise
        assert target.iloc[3] == 1   # 105 -> 120 rises

    def test_target_is_binary(self, parts):
        _, target = parts
        assert set(target.dropna().unique()) <= {0, 1}

    def test_target_aligns_with_the_price_index(self, btc_price):
        assert direction.build_target(btc_price).index.equals(btc_price.index)


class TestFeatureConstruction:
    def test_all_declared_blocks_are_present(self, parts):
        features, _ = parts
        for name in direction.ALL_FEATURES:
            assert name in features.columns, f"missing feature {name}"

    def test_no_feature_uses_future_information(self, btc_price):
        """
        Truncating the series must not change any feature value on the days that
        survive. A feature computed with a forward-looking window would shift.
        """
        full = direction.build_features(btc_price)
        truncated = direction.build_features(btc_price.iloc[:-30])
        common = truncated.index[-50:]
        for column in ("rsi", "return_1d", "return_7d", "drawdown"):
            a = full.loc[common, column]
            b = truncated.loc[common, column]
            assert np.allclose(a.dropna(), b.dropna(), equal_nan=True), (
                f"{column} changed when future data was removed — it is look-ahead"
            )

    def test_missing_feeds_drop_columns_rather_than_failing(self, btc_price):
        """A dead feed must cost its own columns, not the whole model."""
        features = direction.build_features(btc_price, sentiment=None, onchain=None)
        assert not features.empty
        for name in direction.TECHNICAL_FEATURES:
            assert name in features.columns
        for name in direction.SENTIMENT_FEATURES + direction.ONCHAIN_FEATURES:
            assert name not in features.columns

    def test_infinities_are_scrubbed(self, parts):
        features, _ = parts
        assert not np.isinf(features.select_dtypes(include=[np.number])).to_numpy().any()

    def test_short_moving_average_keeps_the_sample_usable(self, btc_price):
        """
        price_vs_ma uses the SHORT window deliberately: the long one would cost
        199 rows of warm-up on a 365-day history and leave too little to
        cross-validate on.
        """
        features = direction.build_features(btc_price)
        assert features["price_vs_ma"].notna().sum() > 250


class TestEvaluation:
    def test_reports_against_the_base_rate_not_fifty_percent(self, parts):
        """
        Accuracy alone is meaningless when one class dominates. The base rate is
        measured on the same held-out days the model was scored on.
        """
        features, target = parts
        result = direction.evaluate_direction_model(features, target, "BTC")
        assert 0.0 <= result.base_rate <= 1.0
        assert result.base_rate >= 0.5, "base rate must be the MAJORITY class"
        assert result.lift == pytest.approx(result.accuracy - result.base_rate)

    def test_accuracy_is_a_probability(self, parts):
        features, target = parts
        result = direction.evaluate_direction_model(features, target, "BTC")
        assert 0.0 <= result.accuracy <= 1.0

    @pytest.mark.parametrize("model_name", ["Logistic regression", "Gradient boosting"])
    def test_both_models_run(self, parts, model_name):
        features, target = parts
        result = direction.evaluate_direction_model(
            features, target, "BTC", model_name=model_name
        )
        assert np.isfinite(result.accuracy)
        assert result.model_name == model_name

    def test_importances_are_reported_for_both_models(self, parts):
        """
        Guards a real bug: GradientBoostingClassifier defines __getitem__, so
        treating that as "this is a Pipeline" indexed into it and pulled out a
        single tree, silently returning no importances at all.
        """
        features, target = parts
        for model_name in ("Logistic regression", "Gradient boosting"):
            result = direction.evaluate_direction_model(
                features, target, "BTC", model_name=model_name
            )
            assert not result.importances.empty, f"{model_name} reported none"
            assert set(result.importances.index) <= set(result.feature_names)

    def test_confusion_matrix_totals_the_test_days(self, parts):
        features, target = parts
        result = direction.evaluate_direction_model(features, target, "BTC")
        assert result.confusion is not None
        assert int(result.confusion.sum()) == result.n_test

    def test_too_little_data_returns_nan_not_a_crash(self):
        idx = pd.date_range("2024-01-01", periods=20, freq="D")
        features = pd.DataFrame({"a": np.arange(20.0)}, index=idx)
        target = pd.Series(np.zeros(20, dtype=int), index=idx, name="direction")
        result = direction.evaluate_direction_model(features, target, "BTC")
        assert np.isnan(result.accuracy)
        assert "Not enough data" in result.verdict()

    def test_single_class_target_returns_nan(self):
        idx = pd.date_range("2024-01-01", periods=200, freq="D")
        rng = np.random.default_rng(0)
        features = pd.DataFrame(rng.normal(size=(200, 3)), index=idx, columns=list("abc"))
        target = pd.Series(np.ones(200, dtype=int), index=idx, name="direction")
        assert np.isnan(
            direction.evaluate_direction_model(features, target, "BTC").accuracy
        )

    def test_unknown_model_name_is_rejected(self):
        with pytest.raises(ValueError):
            direction._make_model("magic")


class TestVerdict:
    @pytest.mark.parametrize(
        "accuracy,base_rate,fragment",
        [
            (0.60, 0.50, "above the"),
            (0.51, 0.50, "inside the noise"),
            (0.45, 0.50, "did not beat"),
        ],
    )
    def test_verdict_matches_the_lift_band(self, accuracy, base_rate, fragment):
        result = direction.DirectionResult(
            symbol="BTC", model_name="Gradient boosting", feature_names=[],
            accuracy=accuracy, base_rate=base_rate, roc_auc=0.5,
            precision=0.5, recall=0.5,
        )
        assert fragment in result.verdict()

    def test_no_lift_is_reported_as_a_result_not_an_error(self):
        result = direction.DirectionResult(
            symbol="BTC", model_name="Gradient boosting", feature_names=[],
            accuracy=0.45, base_rate=0.55, roc_auc=0.5, precision=0.5, recall=0.5,
        )
        assert result.beats_baseline is False
        assert "expected result" in result.verdict()


class TestAblation:
    def test_one_row_per_feature_block(self, parts):
        features, target = parts
        table = direction.feature_block_ablation(features, target, "BTC")
        assert len(table) == 5
        assert "Technical only" in table.index
        assert "All features" in table.index

    def test_every_row_carries_a_comparable_base_rate(self, parts):
        features, target = parts
        table = direction.feature_block_ablation(features, target, "BTC")
        assert table["Base rate"].notna().all()
        assert np.allclose(table["Lift"], table["Accuracy"] - table["Base rate"])
