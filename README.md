# Demand Forecasting & Inventory Optimization

This project demonstrates an end-to-end demand forecasting and inventory optimization workflow:

- Generate synthetic SKU-level daily sales data
- Clean and aggregate the raw sales
- Build ARIMA-based time series forecasts per SKU
- Compute EOQ, safety stock, and reorder points
- Export results for dashboarding in Power BI or Tableau

## Quickstart

```bash
pip install -r requirements.txt
python data/generate_sample_data.py
python src/data_prep.py
python src/forecasting.py
python src/inventory.py
```

Outputs are written into the `data/` folder.
# Demand-Forecasting-Inventory-Optimization
# Supplier-Performance-Risk-Analytics
