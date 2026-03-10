"""
STEP 2: Define the LSTM Model
==============================
An LSTM (Long Short-Term Memory) is a type of neural network that is
excellent at learning patterns in sequential data — like weekly sales
over time. It "remembers" past values to predict future ones.

Think of it like this:
- You look at the last 8 weeks of sales
- The LSTM learns the pattern (trend, seasonality, etc.)
- It predicts what next week's sales will be

This is the same concept used by companies like Amazon, Zara, and
Deliveroo for demand forecasting.
"""

import torch
import torch.nn as nn


class DemandLSTM(nn.Module):
    """
    LSTM model for demand forecasting.

    Architecture:
        Input (features) -> LSTM layers -> Fully Connected -> Output (prediction)

    Parameters:
        input_size:  Number of features per time step (e.g. units_sold, month, etc.)
        hidden_size: Number of LSTM hidden units (how much "memory" the model has)
        num_layers:  Number of stacked LSTM layers (deeper = more complex patterns)
        output_size: Number of values to predict (1 = next week's demand)
    """

    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(DemandLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer — the core of the model
        # batch_first=True means input shape is (batch, sequence, features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2  # Prevents overfitting by randomly dropping 20% of connections
        )

        # Fully connected layers — transform LSTM output into a prediction
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),       # Activation function — introduces non-linearity
            nn.Dropout(0.1), # Additional dropout for regularisation
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        """
        Forward pass — how data flows through the model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
               Example: (32, 8, 3) = 32 samples, 8 weeks history, 3 features

        Returns:
            Prediction tensor of shape (batch_size, output_size)
        """
        # Initialise hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Run input through LSTM
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Take only the last time step's output
        # This contains the LSTM's "summary" of the entire sequence
        last_output = lstm_out[:, -1, :]

        # Pass through fully connected layers to get prediction
        prediction = self.fc(last_output)

        return prediction
