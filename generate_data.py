"""
STEP 1: Generate Synthetic Retail Demand Data
==============================================
This script creates realistic weekly sales data for 6 product categories,
sold across 3 channels and 3 regions, over 3 years (156 weeks) each.
The data includes:
- Seasonal patterns (higher sales in winter/summer, category-specific)
- Channel and region scale effects (e.g. Online running higher volume than Wholesale)
- Weekly trends (slight growth over time)
- Random noise (real-world variability)

This mimics the kind of multi-dimensional data you'd see in a retail planning team
tracking demand by category, channel, and region.
"""

import pandas as pd
import numpy as np
import os

# Set seed so the data is reproducible every time you run it
np.random.seed(42)

# ── Configuration ──
CATEGORIES = [
    "Leather Goods", "Outerwear", "Accessories",
    "Scarves", "Shoes", "Ready to Wear",
]
CHANNELS = ["Online", "In-Store", "Wholesale"]
REGIONS = ["UK", "EU", "US"]

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

# Channel scale effects — relative volume per channel vs. the category base
channel_multiplier = {
    "Online": 1.20,     # highest volume channel
    "In-Store": 1.00,   # baseline
    "Wholesale": 0.55,  # fewer, larger transactions -> lower unit count in this model
}

# Region scale effects — relative market size per region vs. the category base
region_multiplier = {
    "UK": 1.00,   # baseline (home market)
    "EU": 0.90,   # slightly smaller in this dataset
    "US": 1.35,   # largest market
}


def generate_series(category, channel, region, n_weeks, start_date):
    """Generate weekly demand data for a single (category, channel, region) series."""
    dates = pd.date_range(start=start_date, periods=n_weeks, freq="W-MON")
    base = base_demand[category]
    seasonal = seasonal_profiles[category]
    ch_mult = channel_multiplier[channel]
    rg_mult = region_multiplier[region]

    demand = []
    for i, date in enumerate(dates):
        # 1. Seasonal effect — based on which month we're in
        month_idx = date.month - 1
        season_multiplier = seasonal[month_idx]

        # 2. Growth trend — 0.1% weekly growth (compounding)
        trend = 1 + (i * 0.001)

        # 3. Random noise — +/- 15% variability (independent draw per series/week)
        noise = np.random.normal(1.0, 0.15)

        # Calculate final demand (minimum 0 units)
        weekly_demand = max(0, int(base * ch_mult * rg_mult * season_multiplier * trend * noise))
        demand.append(weekly_demand)

    df = pd.DataFrame({
        "date": dates,
        "category": category,
        "channel": channel,
        "region": region,
        "units_sold": demand,
        "week_number": range(1, n_weeks + 1),
        "month": [d.month for d in dates],
        "year": [d.year for d in dates],
    })
    return df


def main():
    """Generate and save the full dataset across all category x channel x region combinations."""
    print("Generating synthetic retail demand data...")
    print(f"  {len(CATEGORIES)} categories x {len(CHANNELS)} channels x {len(REGIONS)} regions "
          f"= {len(CATEGORIES) * len(CHANNELS) * len(REGIONS)} series, {n_weeks} weeks each")

    all_data = []
    for cat in CATEGORIES:
        for ch in CHANNELS:
            for rg in REGIONS:
                series_data = generate_series(cat, ch, rg, n_weeks, start_date)
                all_data.append(series_data)

    df = pd.concat(all_data, ignore_index=True)

    output_path = os.path.join(os.path.dirname(__file__), "data", "weekly_demand.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows to {output_path}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Categories: {df['category'].nunique()} | Channels: {df['channel'].nunique()} | "
          f"Regions: {df['region'].nunique()}")

    return df


if __name__ == "__main__":
    main()
