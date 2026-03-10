# Retail Demand Forecasting Model

An end-to-end machine learning pipeline that predicts weekly product demand using an LSTM neural network, deployed as a Flask REST API with an interactive web dashboard.

## Results

- **Model**: 2-layer LSTM with 64 hidden units and dropout regularisation
- **Test MAPE**: 11.9% (Mean Absolute Percentage Error)
- **Categories**: 6 product categories across 3 channels and 3 regions

## Tech Stack

- **Python** — core language
- **PyTorch** — LSTM neural network implementation
- **Flask** — REST API and web interface
- **NumPy / Pandas** — data processing and feature engineering
- **Matplotlib** — forecast visualisation and training loss curves

## Project Structure

```
├── generate_data.py    # Synthetic retail data generation (936 rows)
├── model.py            # DemandLSTM architecture
├── train.py            # Training pipeline with evaluation
├── app.py              # Flask API + web dashboard
├── trained_model.pth   # Saved model weights
├── scaler.pkl          # Fitted MinMaxScaler
├── forecast_results.png # Prediction vs actual chart
└── data/               # Generated training data
```

## How to Run

```bash
# Install dependencies
pip install torch numpy pandas matplotlib flask scikit-learn

# Generate data
python generate_data.py

# Train the model
python train.py

# Launch the web app
python app.py
# Visit http://localhost:5000
```

## API Endpoint

```
POST /predict
Content-Type: application/json

{
  "features": [1000, 850, 920, 780, 1100, 950, 1020, 890]
}
```

Returns predicted demand for the next period.
