"""
modules/lstm/lstm.py

Trajectory prediction LSTM for TennisTracker.

Models:
    TrajectoryLSTM  — takes a (seq_len, 2) window of (x, y) positions and
                      predicts the next position.

Usage:
    from modules.lstm.train import load_lstm
    model = load_lstm('best')
"""

import torch
import torch.nn as nn


class TrajectoryLSTM(nn.Module):
    """
    Multi-layer LSTM for ball/player trajectory prediction.

    Input:  (batch, seq_len, 2)  — normalised (x, y) in [0, 1]
    Output: (batch, 1, 2)        — predicted next position

    Args:
        input_size  : features per timestep (default 2 -> x, y)
        hidden_size : LSTM hidden units (default 128)
        num_layers  : stacked LSTM layers (default 2)
        dropout     : dropout between LSTM layers (default 0.2)
    """

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 2)
        Returns:
            predictions: (batch, 1, 2)
        """
        projected = self.input_proj(x)           # (batch, seq_len, hidden)
        lstm_out, _ = self.lstm(projected)        # (batch, seq_len, hidden)
        last_hidden = lstm_out[:, -1, :]          # (batch, hidden)
        out = self.decoder(last_hidden)           # (batch, 2)
        return out.unsqueeze(1)                   # (batch, 1, 2)

    def predict_sequence(
        self,
        seed: torch.Tensor,
        steps: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive multi-step prediction.

        Args:
            seed  : (seq_len, 2) or (1, seq_len, 2)
            steps : total future steps to predict
        Returns:
            trajectory: (steps, 2)
        """
        if device is None:
            device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        with torch.no_grad():
            window = seed.to(device)
            if window.dim() == 2:
                window = window.unsqueeze(0)      # (1, seq_len, 2)
            preds = []
            for _ in range(steps):
                pred = self(window)               # (1, 1, 2)
                preds.append(pred)
                window = torch.cat([window[:, 1:], pred], dim=1)
            result = torch.cat(preds, dim=1).squeeze(0)  # (steps, 2)
        if was_training:
            self.train()
        return result