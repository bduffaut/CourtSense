"""
modules/mlp/mlp.py

Fully-connected MLP baseline classifier for TennisTracker.

Input : 1024-d vector  (32×32 grayscale crop, flattened)
Output: 3 logits       (player / ball / background)

This is intentionally a pure MLP — no convolutions — so it can be compared
directly against TennisClassifier (CNN) to demonstrate why spatial feature
extraction matters for image data.

Architecture:
    Linear(1024 → 512) → BatchNorm1d → ReLU → Dropout(0.5)
    Linear(512  → 256) → BatchNorm1d → ReLU → Dropout(0.4)
    Linear(256  → 128) → BatchNorm1d → ReLU → Dropout(0.3)
    Linear(128  → 3)

Parameter count: ~690 K
Expected val accuracy: ~65–75 % (vs CNN ~99 %)
"""

import torch
import torch.nn as nn


CLASSES = ["player", "ball", "background"]
INPUT_DIM = 32 * 32   # 1024 — flattened 32×32 grayscale


class TennisMLP(nn.Module):

    CLASSES = CLASSES

    def __init__(self, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout + 0.1),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout - 0.1),

            nn.Linear(128, len(CLASSES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1024)  — pre-flattened grayscale pixel vector
        Returns:
            logits: (batch, 3)
        """
        return self.net(x)

    def predict(self, x: torch.Tensor) -> list[str]:
        with torch.no_grad():
            indices = self.forward(x).argmax(dim=1)
        return [CLASSES[i] for i in indices.tolist()]
