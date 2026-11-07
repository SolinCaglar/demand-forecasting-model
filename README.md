# Retail Demand Forecasting Model

An end-to-end machine learning pipeline that predicts weekly product demand using an LSTM neural network, deployed as a Flask REST API with an interactive web dashboard.

The data is **synthetically generated** (see `generate_data.py`) to mimic realistic retail patterns — seasonality, trend growth, and noise — rather than pulled from a live business. The point of the project is the pipeline itself: proper time-series validation, a model that generalises across multiple series, and a deployable API.

## Results

- **Model**: 2-layer LSTM (64 hidden units, dropout regularisation), shared across all series
- **Scope**: 6 product categories x 3 channels x 3 regions = 54 individual weekly time series, 3 years (156 weeks) each
- **Test MAPE**: 14.5% (Mean Absolute Percentage Error), evaluated on each series' own held-out *future* weeks (chronological 80/20 split per series — the model never sees test-period data during training)
- Category, channel, and region are passed into the model as one-hot features, so a single shared model learns to forecast any of the 54 combinations rather than one model per category

### Earlier iteration

An initial version of this project trained a separate model on a single category only, reaching 11.9% MAPE. That version didn't yet generalise across channels or regions. This version extends the pipeline to genuinely cover all 54 category x channel x region series with one shared model — a harder, more general task, which is why the MAPE is higher but the scope is real.

## Tech Stack

- **Python** — core language
- **PyTorch** — LSTM neural network implementation
- **Flask** — REST API and web interface
- **NumPy / Pandas / scikit-learn** — data processing, feature engineering, scaling
- **Matplotlib** — forecast visualisation and training loss curves

## Project Structure

```
├── generate_data.py     # Synthetic retail data generation (8,424 rows: 6 categories x 3 channels x 3 regions x 156 weeks)
├── model.py              # DemandLSTM architecture
├── train.py               # Training pipeline: builds sequences per series, trains one shared model, evaluates on held-out future weeks
├── app.py                 # Flask API + web dashboard (category/channel/region selectable)
├── trained_model.pth    # Saved model weights
├── scaler.pkl             # Fitted global MinMaxScaler
├── forecast_results.png # Prediction vs actual chart (example series) + training loss curve
└── data/                   # Generated training data
```

## How to Run

```bash
# Install dependencies
pip install torch numpy pandas matplotlib flask scikit-learn

# Generate data (all 54 category x channel x region series)
python generate_data.py

# Train the shared model and evaluate on held-out future weeks
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
  "category": "Outerwear",
  "channel": "Online",
  "region": "UK",
  "recent_sales": [180, 195, 210, 200, 185, 190, 205, 215],
  "month": 3
}
```

Returns the predicted demand for that specific category/channel/region series for the next week.

**Categories**: Leather Goods, Outerwear, Accessories, Scarves, Shoes, Ready to Wear
**Channels**: Online, In-Store, Wholesale
**Regions**: UK, EU, US
