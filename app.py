"""
STEP 4: Deploy as a Web API
=============================
This Flask app serves the trained model as a REST API.
You can send it recent sales data and it returns a demand prediction.

To run:
    python app.py

Then open your browser to:
    http://localhost:5000

API endpoint:
    POST /predict
    Send JSON with recent weekly sales, get back a prediction.

Example request (using curl):
    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"recent_sales": [180, 195, 210, 200, 185, 190, 205, 215], "month": 3}'
"""

import os
import pickle
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template_string

from model import DemandLSTM

BASE_DIR = os.path.dirname(__file__)

app = Flask(__name__)

# ── Load the trained model and scaler ──
def load_model():
    """Load the trained LSTM model and scaler."""
    # Load scaler (to normalise/denormalise values)
    with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    # Recreate model architecture and load trained weights
    model = DemandLSTM(input_size=3, hidden_size=64, num_layers=2, output_size=1)
    model.load_state_dict(torch.load(
        os.path.join(BASE_DIR, "trained_model.pth"),
        map_location=torch.device("cpu"),
        weights_only=True
    ))
    model.eval()  # Set to evaluation mode

    return model, scaler


model, scaler = load_model()


# ── HTML template for the web interface ──
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Demand Forecasting API</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #f5f5f5; color: #1a1a1a; }
        .container { max-width: 800px; margin: 40px auto; padding: 0 20px; }
        h1 { font-size: 28px; margin-bottom: 8px; }
        .subtitle { color: #666; margin-bottom: 30px; font-size: 14px; }
        .card { background: white; border-radius: 12px; padding: 30px; margin-bottom: 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        .card h2 { font-size: 18px; margin-bottom: 15px; }
        label { display: block; font-weight: 600; margin-bottom: 6px; font-size: 14px; }
        input, select { width: 100%; padding: 10px 12px; border: 1px solid #ddd; border-radius: 8px;
                        font-size: 14px; margin-bottom: 15px; }
        input:focus, select:focus { outline: none; border-color: #2563eb; }
        button { background: #1a1a1a; color: white; padding: 12px 24px; border: none;
                 border-radius: 8px; font-size: 14px; cursor: pointer; font-weight: 600; }
        button:hover { background: #333; }
        .result { margin-top: 20px; padding: 20px; background: #f0f7ff; border-radius: 8px;
                  border-left: 4px solid #2563eb; display: none; }
        .result h3 { color: #2563eb; margin-bottom: 8px; }
        .result .number { font-size: 36px; font-weight: 700; color: #1a1a1a; }
        .result .detail { color: #666; font-size: 13px; margin-top: 8px; }
        .help { font-size: 12px; color: #999; margin-top: -10px; margin-bottom: 15px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Demand Forecasting Model</h1>
        <p class="subtitle">LSTM Neural Network | PyTorch | Retail Demand Prediction</p>

        <div class="card">
            <h2>Predict Next Week's Demand</h2>
            <p style="color:#666; font-size:13px; margin-bottom:20px;">
                Enter the last 8 weeks of unit sales and the target month to forecast next week's demand.
            </p>

            <label>Last 8 Weeks of Sales (comma-separated)</label>
            <input type="text" id="sales" placeholder="e.g. 180, 195, 210, 200, 185, 190, 205, 215"
                   value="180, 195, 210, 200, 185, 190, 205, 215">
            <p class="help">Enter 8 numbers representing weekly unit sales</p>

            <label>Target Month</label>
            <select id="month">
                <option value="1">January</option><option value="2">February</option>
                <option value="3" selected>March</option><option value="4">April</option>
                <option value="5">May</option><option value="6">June</option>
                <option value="7">July</option><option value="8">August</option>
                <option value="9">September</option><option value="10">October</option>
                <option value="11">November</option><option value="12">December</option>
            </select>

            <button onclick="predict()">Predict Demand</button>

            <div class="result" id="result">
                <h3>Predicted Demand</h3>
                <div class="number" id="prediction"></div>
                <div class="detail" id="detail"></div>
            </div>
        </div>

        <div class="card">
            <h2>API Usage</h2>
            <p style="font-size: 13px; color: #666; line-height: 1.6;">
                <strong>POST /predict</strong><br>
                <code style="background:#f5f5f5; padding:2px 6px; border-radius:4px;">
                curl -X POST http://localhost:5000/predict
                -H "Content-Type: application/json"
                -d '{"recent_sales": [180, 195, 210, 200, 185, 190, 205, 215], "month": 3}'
                </code>
            </p>
        </div>
    </div>

    <script>
        async function predict() {
            const salesInput = document.getElementById('sales').value;
            const month = parseInt(document.getElementById('month').value);
            const sales = salesInput.split(',').map(s => parseInt(s.trim()));

            if (sales.length !== 8 || sales.some(isNaN)) {
                alert('Please enter exactly 8 valid numbers');
                return;
            }

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ recent_sales: sales, month: month })
            });

            const data = await response.json();
            document.getElementById('prediction').textContent = data.predicted_demand + ' units';
            document.getElementById('detail').textContent =
                'Based on ' + sales.length + ' weeks of history | Avg input: ' +
                Math.round(sales.reduce((a,b) => a+b) / sales.length) + ' units/week';
            document.getElementById('result').style.display = 'block';
        }
    </script>
</body>
</html>
"""


@app.route("/")
def home():
    """Serve the web interface."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict next week's demand.

    Expected JSON:
        {
            "recent_sales": [180, 195, 210, 200, 185, 190, 205, 215],
            "month": 3
        }

    Returns:
        {
            "predicted_demand": 198,
            "input_avg": 196.9,
            "month": 3,
            "model": "LSTM (2-layer, 64 hidden units)"
        }
    """
    data = request.get_json()

    # Validate input
    recent_sales = data.get("recent_sales", [])
    month = data.get("month", 1)

    if len(recent_sales) != 8:
        return jsonify({"error": "Please provide exactly 8 weeks of sales data"}), 400

    # Prepare input features
    # Normalise sales using the same scaler used during training
    sales_scaled = scaler.transform(np.array(recent_sales).reshape(-1, 1)).flatten()

    # Calculate month encoding (sin/cos)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    # Build the input sequence: (1, 8, 3) — 1 sample, 8 timesteps, 3 features
    sequence = np.zeros((8, 3))
    for i in range(8):
        sequence[i, 0] = sales_scaled[i]  # Normalised sales
        sequence[i, 1] = month_sin         # Month sin
        sequence[i, 2] = month_cos         # Month cos

    # Convert to PyTorch tensor
    input_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        prediction_scaled = model(input_tensor)

    # Convert back to original scale
    prediction = scaler.inverse_transform(prediction_scaled.numpy())[0][0]
    predicted_demand = max(0, int(round(prediction)))

    return jsonify({
        "predicted_demand": predicted_demand,
        "input_avg": round(float(np.mean(recent_sales)), 1),
        "month": month,
        "model": "LSTM (2-layer, 64 hidden units)"
    })


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model_loaded": True})


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("DEMAND FORECASTING API")
    print("=" * 50)
    print("Open http://localhost:5000 in your browser")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000)
