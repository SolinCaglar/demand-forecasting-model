"""
STEP 1: Generate Synthetic Retail Demand Data
==============================================
This script creates realistic weekly sales data for 6 product categories
across 3 years (156 weeks). The data includes:
- Seasonal patterns (higher sales in winter/summer)
- Weekly trends (slight growth over time)
- Random noise (real-world variability)
- Category-specific behaviour

This mimics the kind of data you'd see in a retail planning team.
"""

import pandas as pd
import numpy as np
import os

# Set seed so the data is reproducible every time you run it
np.random.seed(42)

# ── Configuration ──
categories = [
    "Leather Goods", "Outerwear", "Accessories",
    "Scarves", "Shoes", "Ready to Wear"
]
n_weeks = 156  # 3 years of weekly data
start_date = pd.Timestamp("2023-01-02")  # Start on a Monday

# Base weekly demand for each category (units sold per week)
base_demand = {
    "Leather Goods": 120,
    "Outerwear": 200,
    "Accessories": 300,
    "Scarves": 150,
    "Shoes": 180,
    "Ready to Wear": 250,
}

# Seasonal multipliers — which months are strong for each category
# Values > 1 mean higher demand, < 1 mean lower demand
seasonal_profiles = {
    "Leather Goods":   [0.8, 0.8, 0.9, 1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 1.2, 1.3, 1.4],
    "Outerwear":       [1.3, 1.2, 1.0, 0.7, 0.5, 0.4, 0.4, 0.5, 0.8, 1.1, 1.4, 1.5],
    "Accessories":     [1.0, 0.9, 1.0, 1.0, 1.1, 1.2, 1.1, 1.0, 1.0, 1.0, 1.2, 1.3],
    "Scarves":         [1.5, 1.3, 1.0, 0.6, 0.4, 0.3, 0.3, 0.4, 0.7, 1.1, 1.4, 1.6],
    "Shoes":           [0.8, 0.8, 1.0, 1.1, 1.2, 1.3, 1.2, 1.1, 1.0, 0.9, 0.9, 1.0],
    "Ready to Wear":   [0.9, 0.9, 1.0, 1.1, 1.1, 1.2, 1.1, 1.0, 1.1, 1.0, 1.1, 1.2],
}


def generate_category_data(category, n_weeks, start_date):
    """Generate weekly demand data for a single category."""
    dates = pd.date_range(start=start_date, periods=n_weeks, freq="W-MON")
    base = base_demand[category]
    seasonal = seasonal_profiles[category]

    demand = []
    for i, date in enumerate(dates):
        # 1. Seasonal effect — based on which month we're in
        month_idx = date.month - 1
        season_multiplier = seasonal[month_idx]

        # 2. Growth trend — 0.1% weekly growth (compounding)
        trend = 1 + (i * 0.001)

        # 3. Random noise — +/- 15% variability
        noise = np.random.normal(1.0, 0.15)

        # Calculate final demand (minimum 0 units)
        weekly_demand = max(0, int(base * season_multiplier * trend * noise))
        demand.append(weekly_demand)

    df = pd.DataFrame({
        "date": dates,
        "category": category,
        "units_sold": demand,
        "week_number": range(1, n_weeks + 1),
        "month": [d.month for d in dates],
        "year": [d.year for d in dates],
    })
    return df


def main():
    """Generate and save the full dataset."""
    print("Generating synthetic retail demand data...")

    # Generate data for all categories
    all_data = []
    for cat in categories:
        cat_data = generate_category_data(cat, n_weeks, start_date)
        all_data.append(cat_data)
        print(f"  {cat}: {len(cat_data)} weeks, avg demand = {cat_data['units_sold'].mean():.0f} units/week")

    # Combine into a single DataFrame
    df = pd.concat(all_data, ignore_index=True)

    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), "data", "weekly_demand.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows to {output_path}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Categories: {df['category'].nunique()}")

    return df


if __name__ == "__main__":
    main()
