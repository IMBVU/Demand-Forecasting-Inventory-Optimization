import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


def train_test_split_time_series(df: pd.DataFrame, test_size: int
                                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return df.iloc[:-test_size], df.iloc[-test_size:]


def fit_arima_for_sku(df_sku: pd.DataFrame, order=(1, 1, 1)):
    """Fit a SARIMAX(ARIMA) model for a single SKU."""
    ts = df_sku.set_index("date")["daily_units"].asfreq("D").fillna(0)
    model = SARIMAX(ts, order=order, enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    return results


def forecast_sku(df_sku: pd.DataFrame, horizon: int = 90, order=(1, 1, 1)) -> pd.DataFrame:
    sku = df_sku["sku"].iloc[0]
    model_res = fit_arima_for_sku(df_sku, order=order)
    forecast = model_res.get_forecast(steps=horizon)
    fc_series = forecast.predicted_mean
    fc_index = fc_series.index

    out = pd.DataFrame({
        "sku": sku,
        "date": fc_index,
        "forecast_units": fc_series.values
    })
    return out


def batch_forecast(df: pd.DataFrame, horizon: int = 90, min_history: int = 90) -> pd.DataFrame:
    sku_forecasts = []
    for sku, group in df.groupby("sku"):
        if len(group) < min_history:
            continue
        fc = forecast_sku(group, horizon=horizon)
        sku_forecasts.append(fc)

    if not sku_forecasts:
        return pd.DataFrame(columns=["sku", "date", "forecast_units"])

    return pd.concat(sku_forecasts, ignore_index=True)


if __name__ == "__main__":
    daily = pd.read_csv("data/clean_daily_sales.csv", parse_dates=["date"])
    forecasts = batch_forecast(daily, horizon=180)
    forecasts.to_csv("data/sku_forecasts.csv", index=False)
    print("Wrote data/sku_forecasts.csv")
