"""
The machine-learning tab.

Two of this project's four feeds — Fear & Greed and the Bitcoin on-chain series —
were collected on every run and then only displayed. This tab settles whether
they carry any information about tomorrow, by training a classifier on them and
scoring it against the one baseline that matters: always predicting the majority
class.

The tab is written to be able to report failure. On a near-random-walk series the
honest answer is usually "no lift", and a dashboard that can only report success
is not measuring anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import config
from src.analysis import direction

MODELS = ["Gradient boosting", "Logistic regression"]


@st.cache_data(show_spinner="Cross-validating the direction model…", ttl=60 * 60)
def _evaluate(
    _features: pd.DataFrame,
    _target: pd.Series,
    symbol: str,
    model_name: str,
    days: int,
) -> direction.DirectionResult:
    """
    Underscored arguments are excluded from Streamlit's cache key, so ``symbol``,
    ``model_name`` and ``days`` have to be passed explicitly — otherwise changing
    the history window would hand back the previous window's model.
    """
    return direction.evaluate_direction_model(
        _features, _target, symbol=symbol, model_name=model_name
    )


@st.cache_data(show_spinner="Running the feature-block ablation…", ttl=60 * 60)
def _ablate(
    _features: pd.DataFrame,
    _target: pd.Series,
    symbol: str,
    model_name: str,
    days: int,
) -> pd.DataFrame:
    return direction.feature_block_ablation(
        _features, _target, symbol=symbol, model_name=model_name
    )


def render(data, symbol: str, days: int) -> None:
    """Draw the tab for one asset."""
    st.subheader(f"Next-day direction model — {symbol}")
    st.markdown(
        "Can tomorrow's direction be predicted from today's technicals, market "
        "sentiment and on-chain activity? This trains a classifier on exactly "
        "that question and scores it against always predicting the majority class."
    )

    price = data.crypto_frames[symbol]["price"]
    features = direction.build_features(
        price,
        sentiment=data.sentiment,
        onchain=data.onchain if symbol == "BTC" else None,
        macro_prices=data.macro_prices,
    )
    target = direction.build_target(price)

    if symbol != "BTC":
        st.caption(
            "On-chain features are Bitcoin-only — the free Blockchain.info charts "
            "cover no other asset — so they are excluded for this symbol."
        )

    model_name = st.radio(
        "Model", MODELS, horizontal=True,
        help="Logistic regression is linear and readable; gradient boosting can "
             "find interactions between features.",
    )

    result = _evaluate(features, target, symbol, model_name, days)

    if not np.isfinite(result.accuracy):
        st.warning(
            "Not enough overlapping history to cross-validate a model on this "
            "window. Select a longer history window in the sidebar."
        )
        return

    metric = st.columns(4)
    metric[0].metric("Accuracy", f"{result.accuracy:.1%}")
    metric[1].metric("Base rate", f"{result.base_rate:.1%}", help="Always predict the majority class.")
    metric[2].metric(
        "Lift over baseline",
        f"{result.lift:+.1%}",
        "beats baseline" if result.beats_baseline else "no better than baseline",
        delta_color="normal" if result.beats_baseline else "off",
    )
    metric[3].metric(
        "ROC AUC",
        "—" if not np.isfinite(result.roc_auc) else f"{result.roc_auc:.3f}",
        help="0.5 is a coin flip.",
    )

    st.info(result.verdict())

    left, right = st.columns(2)

    with left:
        st.write("**Which features the model leant on**")
        if result.importances.empty:
            st.caption("This model exposes no feature importances.")
        else:
            top = result.importances.head(10).sort_values()
            st.bar_chart(top, horizontal=True, color=config.ACCENT)
            st.caption(
                "Gradient boosting reports impurity gain; logistic regression "
                "reports the absolute standardised coefficient. Neither is "
                "evidence of causation, and on a model with no lift they mostly "
                "describe which noise the model happened to fit."
            )

    with right:
        st.write("**Confusion matrix, pooled across folds**")
        if result.confusion is None:
            st.caption("Not available.")
        else:
            matrix = pd.DataFrame(
                result.confusion,
                index=["Actual: down", "Actual: up"],
                columns=["Predicted: down", "Predicted: up"],
            )
            st.dataframe(matrix, width="stretch")
            st.caption(
                f"Precision {result.precision:.1%} · recall {result.recall:.1%} "
                f"on {result.n_test:,} held-out days."
            )

    st.markdown("---")
    st.write("**Were the previously-unused feeds worth collecting?**")
    st.caption(
        "The same model trained on technical features alone, then with each "
        "additional block of features added. This is the direct test of whether "
        "the Fear & Greed and on-chain feeds carry anything — and it is built to "
        "be able to answer no."
    )

    ablation = _ablate(features, target, symbol, model_name, days)
    if ablation.empty:
        st.info("Not enough data to run the ablation on this window.")
    else:
        st.dataframe(
            ablation.style.format(
                {
                    "Accuracy": "{:.1%}",
                    "Base rate": "{:.1%}",
                    "Lift": "{:+.1%}",
                    "ROC AUC": "{:.3f}",
                }
            ),
            width="stretch",
        )

        best = ablation["Lift"].idxmax()
        best_lift = ablation.loc[best, "Lift"]
        if best_lift > 0.02:
            st.caption(
                f"**{best}** carries the largest lift at {best_lift:+.1%}. On a "
                "sample this size treat that as a hypothesis worth more data, "
                "not a finding."
            )
        else:
            st.caption(
                "No feature block produces a lift worth acting on. That is a "
                "real result: it says next-day direction in this asset is not "
                "recoverable from these inputs, which is exactly what efficient- "
                "market intuition predicts and what most published attempts find "
                "once they validate honestly."
            )

    with st.expander("How this is validated"):
        st.markdown(
            f"""
1. **Features are lagged.** Every predictor is computed from information
   available at the close of day *t*; the target is the sign of the move from
   *t* to *t+1*. Nothing in the feature matrix comes from the day being predicted.
2. **Splits are time-ordered.** `TimeSeriesSplit` trains on the past and tests on
   the future, always. A random split on a time series lets the model learn from
   days that had not happened yet — the single most common way a financial
   machine-learning result turns out to be fictional.
3. **Predictions are pooled across folds** before scoring, so the accuracy covers
   one continuous out-of-sample stretch rather than an average of small,
   noisy fold scores.
4. **The baseline is the majority class**, measured on the same held-out days.
   An asset that rose on 54% of days makes 54% the number to beat, not 50%.
   Accuracy alone would look respectable while adding nothing.

Feature blocks: **technical** ({len(direction.TECHNICAL_FEATURES)}) from price
alone, **sentiment** ({len(direction.SENTIMENT_FEATURES)}) from Fear & Greed,
**on-chain** ({len(direction.ONCHAIN_FEATURES)}) from Bitcoin network activity,
and **macro** ({len(direction.MACRO_FEATURES)}) from equities, the dollar and rates.

A lift at or below zero is not a bug. Daily crypto direction is close to
unpredictable, and a model honest enough to say so is a more useful deliverable
than one reporting 95% accuracy because it leaked.
            """
        )
