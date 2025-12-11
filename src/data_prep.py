import pandas as pd


def load_sales_data(path: str) -> pd.DataFrame:
    """Loads raw sales data from CSV."""
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop bad rows and sort."""
    df = df.dropna(subset=["date", "sku", "units_sold"])
    df = df[df["units_sold"] >= 0]
    df = df.sort_values(["sku", "date"])
    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to daily units per SKU."""
    agg = (
        df.groupby(["sku", "date"], as_index=False)["units_sold"]
        .sum()
        .rename(columns={"units_sold": "daily_units"})
    )
    return agg


if __name__ == "__main__":
    raw = load_sales_data("data/raw_sales.csv")
    clean = clean_sales_data(raw)
    daily = aggregate_daily(clean)
    daily.to_csv("data/clean_daily_sales.csv", index=False)
    print("Wrote data/clean_daily_sales.csv")

