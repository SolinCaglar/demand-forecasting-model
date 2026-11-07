"""
STEP 3: Train the Model
========================
This script:
1. Loads the weekly demand data across ALL category x channel x region series
2. Prepares it for the LSTM (creates sequences of past weeks, one-hot encodes
   category/channel/region as additional per-timestep features)
3. Splits into training and test sets — chronologically, per series
4. Trains a single shared model to predict next week's demand for any
   category/channel/region combination
5. Evaluates accuracy on the held-out (future) weeks and saves the trained model
6. Generates visualisation charts

After running this, you'll have:
- A trained model saved as 'trained_model.pth'
- A chart showing predicted vs actual demand for one example series
- Performance metrics (MAE, RMSE, MAPE) computed across ALL series' held-out weeks
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from model import DemandLSTM

# ── Configuration ──
CATEGORIES = [
    "Leather Goods", "Outerwear", "Accessories",
    "Scarves", "Shoes", "Ready to Wear",
]
CHANNELS = ["Online", "In-Store", "Wholesale"]
REGIONS = ["UK", "EU", "US"]

SEQUENCE_LENGTH = 8   # Look at the last 8 weeks to predict the next week
BATCH_SIZE = 32       # Number of samples processed before the model updates
EPOCHS = 60           # Number of times the model sees the entire training data
LEARNING_RATE = 0.001 # How fast the model learns (too high = unstable, too low = slow)
HIDDEN_SIZE = 64      # LSTM memory capacity
NUM_LAYERS = 2        # Number of stacked LSTM layers
TRAIN_FRACTION = 0.8  # Per-series chronological train/test split

BASE_DIR = os.path.dirname(__file__)


def one_hot(value, categories):
    vec = np.zeros(len(categories))
    vec[categories.index(value)] = 1.0
    return vec


def load_data():
    data_path = os.path.join(BASE_DIR, "data", "weekly_demand.csv")
    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.sort_values(["category", "channel", "region", "date"]).reset_index(drop=True)

    # Fit ONE global scaler across every series, so all series share the same scale
    scaler = MinMaxScaler()
    df["units_scaled"] = scaler.fit_transform(df[["units_sold"]])

    # Cyclical month encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df, scaler


def build_series_features(group):
    """
    Build the per-timestep feature matrix for one (category, channel, region) series.

    Features per timestep (15 total):
        units_scaled, month_sin, month_cos,                 (3)
        category one-hot (6), channel one-hot (3), region one-hot (3)
    The category/channel/region one-hot is constant across the series, but is
    repeated at every timestep so the LSTM sees it alongside the time-varying signal.
    """
    cat_vec = one_hot(group["category"].iloc[0], CATEGORIES)
    ch_vec = one_hot(group["channel"].iloc[0], CHANNELS)
    rg_vec = one_hot(group["region"].iloc[0], REGIONS)
    static = np.concatenate([cat_vec, ch_vec, rg_vec])  # (12,)

    n = len(group)
    time_varying = group[["units_scaled", "month_sin", "month_cos"]].values  # (n, 3)
    static_repeated = np.tile(static, (n, 1))  # (n, 12)

    features = np.concatenate([time_varying, static_repeated], axis=1)  # (n, 15)
    targets = group["units_scaled"].values
    return features, targets


def create_sequences(features, targets, seq_length=SEQUENCE_LENGTH):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i + seq_length])
        y.append(targets[i + seq_length])
    return np.array(X), np.array(y)


def build_dataset(df):
    """
    Build train/test sequence arrays across ALL series.

    Each series is split chronologically on its OWN sequences (80% earliest / 20% most
    recent) before being pooled into the global train/test arrays. This keeps the
    no-future-leakage guarantee per series while giving one shared model to train on.
    """
    X_train_all, y_train_all, X_test_all, y_test_all = [], [], [], []
    example_series = None  # keep one series aside for the visualisation chart

    for (cat, ch, rg), group in df.groupby(["category", "channel", "region"], sort=False):
        group = group.sort_values("date")
        features, targets = build_series_features(group)
        X, y = create_sequences(features, targets)

        split_idx = int(len(X) * TRAIN_FRACTION)
        X_train_all.append(X[:split_idx])
        y_train_all.append(y[:split_idx])
        X_test_all.append(X[split_idx:])
        y_test_all.append(y[split_idx:])

        if cat == "Outerwear" and ch == "Online" and rg == "UK":
            example_series = {
                "dates": group["date"].values,
                "split_idx": split_idx,
                "X_test": X[split_idx:],
                "y_test": y[split_idx:],
            }

    X_train = np.concatenate(X_train_all, axis=0)
    y_train = np.concatenate(y_train_all, axis=0)
    X_test = np.concatenate(X_test_all, axis=0)
    y_test = np.concatenate(y_test_all, axis=0)

    return X_train, y_train, X_test, y_test, example_series


def train_model():
    """Main training function."""
    print("=" * 60)
    print("DEMAND FORECASTING MODEL — TRAINING (multi-series)")
    print("=" * 60)

    print("\n1. Loading data...")
    df, scaler = load_data()
    n_series = df.groupby(["category", "channel", "region"]).ngroups
    print(f"   Loaded {len(df)} rows across {n_series} (category, channel, region) series")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    print("\n2. Building sequences per series and splitting chronologically...")
    X_train, y_train, X_test, y_test, example_series = build_dataset(df)
    print(f"   Training sequences: {len(X_train)}")
    print(f"   Test sequences:     {len(X_test)}")
    print(f"   Features per timestep: {X_train.shape[2]} "
          f"(units_scaled, month_sin, month_cos, category[6], channel[3], region[3])")

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    input_size = X_train.shape[2]
    model = DemandLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=1
    )

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n3. Model architecture:")
    print(f"   Input size:   {input_size}")
    print(f"   Hidden size:  {HIDDEN_SIZE}")
    print(f"   LSTM layers:  {NUM_LAYERS}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    print(f"\n4. Training for {EPOCHS} epochs across all series...")
    train_losses = []
    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for batch_X, batch_y in train_loader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(BASE_DIR, "trained_model.pth"))

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3d}/{EPOCHS} — Loss: {avg_loss:.6f}")

    print(f"\n5. Evaluating on held-out test weeks across all {n_series} series...")
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_t)

    pred_original = scaler.inverse_transform(test_predictions.numpy())
    actual_original = scaler.inverse_transform(y_test_t.numpy())

    mae = np.mean(np.abs(pred_original - actual_original))
    rmse = np.sqrt(np.mean((pred_original - actual_original) ** 2))
    mape = np.mean(np.abs((actual_original - pred_original) / np.maximum(actual_original, 1))) * 100

    print(f"\n   RESULTS (across all {len(X_test)} held-out test sequences, all series):")
    print(f"   ─────────────────────────────────")
    print(f"   MAE  (Mean Absolute Error):  {mae:.1f} units")
    print(f"   RMSE (Root Mean Sq Error):   {rmse:.1f} units")
    print(f"   MAPE (Mean Abs % Error):     {mape:.1f}%")
    print(f"   ─────────────────────────────────")

    if mape < 15:
        print(f"   Model accuracy is GOOD (MAPE < 15%)")
    elif mape < 25:
        print(f"   Model accuracy is ACCEPTABLE (MAPE < 25%)")
    else:
        print(f"   Model accuracy NEEDS IMPROVEMENT (MAPE > 25%)")

    print(f"\n6. Generating charts (example series: Outerwear / Online / UK)...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    if example_series is not None:
        with torch.no_grad():
            ex_pred_scaled = model(torch.FloatTensor(example_series["X_test"]))
        ex_pred = scaler.inverse_transform(ex_pred_scaled.numpy())
        ex_actual = scaler.inverse_transform(example_series["y_test"].reshape(-1, 1))

        test_start = example_series["split_idx"] + SEQUENCE_LENGTH
        test_dates = example_series["dates"][test_start:test_start + len(ex_pred)]

        axes[0].plot(test_dates, ex_actual, label="Actual Demand", color="#2563eb", linewidth=2)
        axes[0].plot(test_dates, ex_pred, label="Predicted Demand", color="#dc2626", linewidth=2, linestyle="--")
        axes[0].fill_between(
            test_dates,
            (ex_pred * 0.85).flatten(),
            (ex_pred * 1.15).flatten(),
            alpha=0.15, color="#dc2626", label="Prediction Range (+/-15%)"
        )
        axes[0].set_title("Demand Forecast: Predicted vs Actual (Outerwear / Online / UK)",
                           fontsize=14, fontweight="bold")
        axes[0].set_ylabel("Units Sold")
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_losses, color="#2563eb", linewidth=2)
    axes[1].set_title("Training Loss Over Epochs (all series)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss (MSE)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(BASE_DIR, "forecast_results.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved chart to {chart_path}")

    import pickle
    with open(os.path.join(BASE_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE")
    print(f"Model saved to: trained_model.pth")
    print(f"Final test MAPE: {mape:.1f}%")
    print(f"{'=' * 60}")

    return mae, rmse, mape, n_series, len(X_train), len(X_test)


if __name__ == "__main__":
    train_model()
