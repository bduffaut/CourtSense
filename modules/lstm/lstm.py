import torch
import torch.nn as nn


class BallTrajectoryLSTM(nn.Module):
    # Predicts the next ball position from a sequence of 8.
    # Input:  (B, seq_len, 2) — normalised (x,y) positions
    # Output: (B, 2)          — predicted next absolute position

    def __init__(
        self,
        hidden_size: int   = 64,
        num_layers:  int   = 2,
        dropout:     float = 0.3,
    ):
        super().__init__()
        # input_size=4: each timestep gets (x, y, dx, dy)
        self.lstm = nn.LSTM(
            input_size  = 4,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    @staticmethod
    def _add_velocity(x: torch.Tensor) -> torch.Tensor:
        vel = torch.zeros_like(x)
        vel[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        return torch.cat([x, vel], dim=-1)  # (B, seq_len, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns displacement from last position, not absolute coords
        x4     = self._add_velocity(x)
        out, _ = self.lstm(x4)
        return self.head(out[:, -1, :])

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            delta = self.forward(x)
            return x[:, -1, :] + delta  # add delta to last known position
