"""
Time-series modelling and price forecasting.

Fits an ARIMA model to the log-price series. Modelling the logarithm keeps
forecasts positive and stabilises variance, and differencing once (d=1) means
the model is really working on log returns — the stationary quantity.

The validation is the part that matters, and it changed substantially from the
first version.

*One-step-ahead, not one long shot.* Previously the model was fitted on the
training split and asked for a single forecast covering the entire test period,
roughly 55 days ahead. For a near-random-walk series that forecast is an almost
flat line, so the resulting error measured how far the price happened to drift
over two months rather than anything about the model. Validation now walks
forward one day at a time, revealing each actual observation before predicting
the next — which is what a person using this daily would actually experience.

*A baseline to measure against.* An RMSE of "$4,200" means nothing on its own.
Every result is now reported alongside the naive random-walk forecast (tomorrow's
price is today's price) and a skill score. Positive skill means the model beats
the naive rule; zero or negative means it does not, which for crypto is the
honest and expected outcome most of the time.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config


@dataclass
class ForecastResult:
    """Everything the dashboard needs to present a forecast honestly."""

    symbol: str
    order: tuple[int, int, int]

    forecast: pd.Series          # point forecast, price scale
    lower: pd.Series             # 80% interval, lower bound
    upper: pd.Series             # 80% interval, upper bound

    # Walk-forward, one-step-ahead accuracy on the held-out test period.
    rmse: float
    mae: float
    mape: float

    # The same metrics for the naive random-walk rule, on the same days.
    naive_rmse: float
    naive_mae: float
    naive_mape: float

    fitted_test: pd.Series = field(default_factory=pd.Series)
    test_actual: pd.Series = field(default_factory=pd.Series)

    @property
    def skill(self) -> float:
        """
        Skill score against the random walk: 1 - RMSE_model / RMSE_naive.

        Above 0 the model adds something; at or below 0 it does not beat simply
        assuming tomorrow looks like today.
        """
        if not np.isfinite(self.naive_rmse) or self.naive_rmse == 0:
            return float("nan")
        return float(1 - self.rmse / self.naive_rmse)

    @property
    def beats_naive(self) -> bool:
        return bool(np.isfinite(self.skill) and self.skill > 0)

    def verdict(self) -> str:
        """One line a non-technical reader can act on."""
        if not np.isfinite(self.skill):
            return "Not enough held-out data to judge the model against a baseline."
        if self.skill > 0.05:
            return (
                f"ARIMA{self.order} beat the random-walk baseline by "
                f"{self.skill:.1%} on held-out days."
            )
        if self.skill > 0:
            return (
                f"ARIMA{self.order} edged the random-walk baseline by "
                f"{self.skill:.1%} — too small to rely on."
            )
        return (
            f"ARIMA{self.order} did not beat the random-walk baseline. "
            "Read the interval as a range of outcomes, not a price target."
        )


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    """RMSE, MAE and MAPE on the price scale."""
    error = actual - predicted
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    mape = float(np.mean(np.abs(error / actual))) * 100
    return rmse, mae, mape


def _fit_arima(series: pd.Series, order: tuple[int, int, int]):
    from statsmodels.tsa.arima.model import ARIMA

    with warnings.catch_warnings():
        # Scoped rather than global: the previous module-level
        # filterwarnings("ignore") silenced every warning in the whole process.
        warnings.simplefilter("ignore")
        return ARIMA(series, order=order).fit()


def _select_order(train: pd.Series, candidates: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    """Pick the order with the lowest AIC on the training split."""
    best_order, best_aic = (1, 1, 1), np.inf
    for order in candidates:
        try:
            aic = _fit_arima(train, order).aic
            if np.isfinite(aic) and aic < best_aic:
                best_aic, best_order = aic, order
        except Exception:
            continue
    return best_order


def _walk_forward(train: pd.Series, test: pd.Series, order: tuple[int, int, int]) -> pd.Series:
    """
    One-step-ahead predictions across the test period.

    Parameters are estimated once on the training split; each actual observation
    is then appended to the model's state before the next prediction is made.
    ``refit=False`` keeps the coefficients fixed while updating the state, which
    is both the fast option and the honest one — refitting at every step would
    let the evaluation quietly use information from later in the test period.
    """
    model = _fit_arima(train, order)
    predictions: list[float] = []

    for timestamp in test.index:
        try:
            predictions.append(float(model.forecast(steps=1).iloc[0]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = model.append(test.loc[[timestamp]], refit=False)
        except Exception:
            # If the state update fails, fall back to persistence for this step.
            predictions.append(float(test.loc[timestamp]))

    return pd.Series(predictions, index=test.index)


def forecast_price(
    price: pd.Series,
    symbol: str,
    horizon: int = config.FORECAST_HORIZON,
    test_frac: float = config.FORECAST_TEST_FRACTION,
) -> ForecastResult:
    """Fit ARIMA on log-price, validate walk-forward against a naive baseline, forecast ahead."""
    price = price.dropna().asfreq("D").ffill()
    log_price = np.log(price)

    n_test = max(10, int(len(log_price) * test_frac))
    train, test = log_price.iloc[:-n_test], log_price.iloc[-n_test:]

    candidate_orders = [(p, 1, q) for p in (0, 1, 2) for q in (0, 1, 2)]
    order = _select_order(train, candidate_orders)

    # --- Walk-forward evaluation, model vs naive ---------------------------- #
    model_pred_log = _walk_forward(train, test, order)

    # Naive random walk: predict today's value for tomorrow. The first test day
    # is predicted from the last training observation.
    naive_pred_log = test.shift(1)
    naive_pred_log.iloc[0] = train.iloc[-1]

    actual_price = np.exp(test.to_numpy())
    rmse, mae, mape = _metrics(actual_price, np.exp(model_pred_log.to_numpy()))
    naive_rmse, naive_mae, naive_mape = _metrics(actual_price, np.exp(naive_pred_log.to_numpy()))

    # --- Refit on everything and project forward ---------------------------- #
    full = _fit_arima(log_price, order)
    forecast_object = full.get_forecast(steps=horizon)
    mean_log = forecast_object.predicted_mean
    ci_log = forecast_object.conf_int(alpha=0.20)  # 80% interval

    future_index = pd.date_range(
        price.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D"
    )

    # Exponentiating a log-scale forecast gives the median price rather than the
    # mean, and exponentiating the log-scale bounds gives the correct price-scale
    # quantiles because the transform is monotonic.
    forecast = pd.Series(np.exp(mean_log.to_numpy()), index=future_index, name="forecast")
    lower = pd.Series(np.exp(ci_log.iloc[:, 0].to_numpy()), index=future_index, name="lower")
    upper = pd.Series(np.exp(ci_log.iloc[:, 1].to_numpy()), index=future_index, name="upper")

    return ForecastResult(
        symbol=symbol,
        order=order,
        forecast=forecast,
        lower=lower,
        upper=upper,
        rmse=rmse,
        mae=mae,
        mape=mape,
        naive_rmse=naive_rmse,
        naive_mae=naive_mae,
        naive_mape=naive_mape,
        fitted_test=pd.Series(np.exp(model_pred_log.to_numpy()), index=test.index),
        test_actual=pd.Series(actual_price, index=test.index),
    )
