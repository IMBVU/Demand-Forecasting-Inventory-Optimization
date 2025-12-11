import numpy as np
import pandas as pd
from scipy.stats import norm


def compute_annual_demand(df_forecast: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df_forecast.groupby("sku", as_index=False)["forecast_units"]
        .sum()
        .rename(columns={"forecast_units": "annual_demand"})
    )
    agg["daily_demand"] = agg["annual_demand"] / 365.0
    return agg


def eoq(demand, order_cost, holding_cost):
    return np.sqrt((2 * demand * order_cost) / holding_cost)


def safety_stock(avg_daily_demand, std_daily_demand, lead_time_days, service_level=0.95):
    z = norm.ppf(service_level)
    std_lt = std_daily_demand * np.sqrt(lead_time_days)
    return z * std_lt


def reorder_point(avg_daily_demand, lead_time_days, safety_stock_units):
    return avg_daily_demand * lead_time_days + safety_stock_units


def build_inventory_policy(
    forecast_df: pd.DataFrame,
    demand_stats_df: pd.DataFrame,
    order_cost: float = 50.0,
    holding_cost: float = 2.0,
    lead_time_days: int = 14,
    service_level: float = 0.95
) -> pd.DataFrame:
    df = forecast_df.merge(demand_stats_df, on="sku", how="left")

    df["eoq"] = df["annual_demand"].apply(
        lambda d: eoq(d, order_cost=order_cost, holding_cost=holding_cost)
    )

    df["safety_stock"] = df.apply(
        lambda row: safety_stock(
            avg_daily_demand=row["daily_demand"],
            std_daily_demand=row["std_daily_demand"],
            lead_time_days=lead_time_days,
            service_level=service_level,
        ),
        axis=1,
    )

    df["reorder_point"] = df.apply(
        lambda row: reorder_point(
            avg_daily_demand=row["daily_demand"],
            lead_time_days=lead_time_days,
            safety_stock_units=row["safety_stock"],
        ),
        axis=1,
    )

    return df


if __name__ == "__main__":
    forecast = pd.read_csv("data/sku_forecasts.csv", parse_dates=["date"])
    annual = compute_annual_demand(forecast)
    demand_stats = annual[["sku"]].copy()
    # Placeholder for demo; in a real project compute from historical variance
    demand_stats["std_daily_demand"] = 5.0

    policy = build_inventory_policy(annual, demand_stats)
    policy.to_csv("data/inventory_policy.csv", index=False)
    print("Wrote data/inventory_policy.csv")
