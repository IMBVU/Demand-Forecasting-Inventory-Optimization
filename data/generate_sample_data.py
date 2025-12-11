import os
import numpy as np
import pandas as pd

np.random.seed(42)

def generate_sales_data(
    start_date="2023-01-01",
    end_date="2024-12-31",
    skus=("SKU_A", "SKU_B", "SKU_C", "SKU_D"),
    base_demand=(20, 35, 50, 15)
):
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    rows = []

    for sku, base in zip(skus, base_demand):
        seasonal_factor = np.sin(np.linspace(0, 6 * np.pi, len(dates)))  # seasonality
        for i, d in enumerate(dates):
            mu = base + 5 * seasonal_factor[i]
            units = max(0, int(np.random.normal(loc=mu, scale=5)))
            rows.append({"date": d, "sku": sku, "units_sold": units})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_sales_data()
    df.to_csv("data/raw_sales.csv", index=False)
    print("Generated data/raw_sales.csv")
