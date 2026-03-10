"""
STEP 3: Train the Model
========================
This script:
1. Loads the weekly demand data
2. Prepares it for the LSTM (creates sequences of past weeks)
3. Splits into training and test sets
4. Trains the model to predict next week's demand
5. Evaluates accuracy and saves the trained model
6. Generates visualisation charts

After running this, you'll have:
- A trained model saved as 'trained_model.pth'
- A chart showing predicted vs actual demand
- Performance metrics (MAE, RMSE, MAPE)
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
SEQUENCE_LENGTH = 8   # Look at the last 8 weeks to predict the next week
BATCH_SIZE = 32       # Number of samples processed before the model updates
EPOCHS = 100          # Number of times the model sees the entire training data
LEARNING_RATE = 0.001 # How fast the model learns (too high = unstable, too low = slow)
HIDDEN_SIZE = 64      # LSTM memory capacity
NUM_LAYERS = 2        # Number of stacked LSTM layers

BASE_DIR = os.path.dirname(__file__)


def load_and_prepare_data(category="Outerwear"):
    """
    Load data for a single category and prepare features.

    Features used:
    - units_sold (normalised): the main signal
    - month_sin, month_cos: encodes seasonality as a smooth cycle
      (December and January are close together, unlike raw month numbers)
    """
    data_path = os.path.join(BASE_DIR, "data", "weekly_demand.csv")
    df = pd.read_csv(data_path, parse_dates=["date"])

    # Filter to one category
    df = df[df["category"] == category].sort_values("date").reset_index(drop=True)

    # Normalise units_sold to range [0, 1] — neural networks work better with small numbers
    scaler = MinMaxScaler()
    df["units_scaled"] = scaler.fit_transform(df[["units_sold"]])

    # Encode month as cyclical features (sin/cos)
    # This tells the model that December (12) and January (1) are adjacent
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df, scaler


def create_sequences(df, seq_length=SEQUENCE_LENGTH):
    """
    Create input/output sequences for the LSTM.

    Example with seq_length=8:
        Input:  weeks 1-8  -> Output: week 9
        Input:  weeks 2-9  -> Output: week 10
        Input:  weeks 3-10 -> Output: week 11
        ...and so on

    This is called a "sliding window" approach.
    """
    features = df[["units_scaled", "month_sin", "month_cos"]].values
    targets = df["units_scaled"].values

    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i + seq_length])  # 8 weeks of features
        y.append(targets[i + seq_length])      # Next week's demand

    return np.array(X), np.array(y)


def train_model():
    """Main training function."""
    print("=" * 60)
    print("DEMAND FORECASTING MODEL — TRAINING")
    print("=" * 60)

    # 1. Load and prepare data
    print("\n1. Loading data...")
    df, scaler = load_and_prepare_data("Outerwear")
    print(f"   Loaded {len(df)} weeks of data")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"   Avg weekly demand: {df['units_sold'].mean():.0f} units")

    # 2. Create sequences
    print("\n2. Creating sequences...")
    X, y = create_sequences(df)
    print(f"   Total sequences: {len(X)}")
    print(f"   Each sequence: {SEQUENCE_LENGTH} weeks -> predict 1 week")
    print(f"   Features per timestep: {X.shape[2]} (units_scaled, month_sin, month_cos)")

    # 3. Split into train (80%) and test (20%)
    # IMPORTANT: We split chronologically, not randomly!
    # (You can't use future data to predict the past)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"\n3. Train/Test split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples:     {len(X_test)}")

    # 4. Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 5. Initialise model, loss function, and optimiser
    input_size = X_train.shape[2]  # Number of features (3)
    model = DemandLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=1
    )

    criterion = nn.MSELoss()  # Mean Squared Error — penalises large prediction errors
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n4. Model architecture:")
    print(f"   Input size:   {input_size}")
    print(f"   Hidden size:  {HIDDEN_SIZE}")
    print(f"   LSTM layers:  {NUM_LAYERS}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # 6. Training loop
    print(f"\n5. Training for {EPOCHS} epochs...")
    train_losses = []
    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()  # Set model to training mode
        epoch_loss = 0

        for batch_X, batch_y in train_loader:
            # Forward pass — make predictions
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

            # Backward pass — calculate gradients
            optimiser.zero_grad()
            loss.backward()

            # Update weights
            optimiser.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(BASE_DIR, "trained_model.pth"))

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3d}/{EPOCHS} — Loss: {avg_loss:.6f}")

    # 7. Evaluate on test set
    print(f"\n6. Evaluating on test set...")
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Don't calculate gradients during evaluation
        test_predictions = model(X_test_t)
        test_loss = criterion(test_predictions, y_test_t)

    # Convert predictions back to original scale (undo normalisation)
    pred_original = scaler.inverse_transform(test_predictions.numpy())
    actual_original = scaler.inverse_transform(y_test_t.numpy())

    # Calculate metrics
    mae = np.mean(np.abs(pred_original - actual_original))
    rmse = np.sqrt(np.mean((pred_original - actual_original) ** 2))
    mape = np.mean(np.abs((actual_original - pred_original) / actual_original)) * 100

    print(f"\n   RESULTS:")
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

    # 8. Generate visualisations
    print(f"\n7. Generating charts...")

    # Chart 1: Predicted vs Actual demand
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Get test dates
    test_start = split_idx + SEQUENCE_LENGTH
    test_dates = df["date"].iloc[test_start:test_start + len(pred_original)].values

    axes[0].plot(test_dates, actual_original, label="Actual Demand", color="#2563eb", linewidth=2)
    axes[0].plot(test_dates, pred_original, label="Predicted Demand", color="#dc2626", linewidth=2, linestyle="--")
    axes[0].fill_between(
        test_dates,
        (pred_original * 0.85).flatten(),
        (pred_original * 1.15).flatten(),
        alpha=0.15, color="#dc2626", label="Prediction Range (+/-15%)"
    )
    axes[0].set_title("Demand Forecast: Predicted vs Actual (Outerwear)", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Units Sold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Chart 2: Training loss over epochs
    axes[1].plot(train_losses, color="#2563eb", linewidth=2)
    axes[1].set_title("Training Loss Over Epochs", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss (MSE)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(BASE_DIR, "forecast_results.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved chart to {chart_path}")

    # Save scaler for the API to use later
    import pickle
    with open(os.path.join(BASE_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE")
    print(f"Model saved to: trained_model.pth")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    train_model()
