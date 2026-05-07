# cnn.py
# Two models built on the same convolutional backbone.
# TennisClassifier      — classifies 128x128 crops as player / ball / background
# TennisKeypointDetector — finds 15 court landmark positions in a full frame
# The backbone (4 conv blocks) is shared between both — same feature extractor,
# different heads depending on the task.

import torch
import torch.nn as nn


# Building block

class ConvBlock(nn.Module):
    # One conv layer + batch norm + relu + maxpool.
    # Every block halves the spatial size and changes the channel count.
    # bias=False on the conv because batchnorm's shift parameter does the same job.

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# Shared backbone

class TennisCNNBackbone(nn.Module):
    # Four conv blocks stacked — each doubles channels, halves spatial size.
    # By the end spatial size is H/16 x W/16 with 256 channels.
    # Input  (B, 3, H, W)
    # Output (B, 256, H/16, W/16)

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(3,   32),   # (B, 32,  H/2,  W/2)
            ConvBlock(32,  64),   # (B, 64,  H/4,  W/4)
            ConvBlock(64,  128),  # (B, 128, H/8,  W/8)
            ConvBlock(128, 256),  # (B, 256, H/16, W/16)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# Model 1: patch classifier

class TennisClassifier(nn.Module):
    # Takes a 128x128 crop and classifies it as player, ball, or background.
    # The conv layers extract features, the FC layers at the end do the classification
    # — that's what the course calls the "MLP head".
    # Classes:
    #   0 — player
    #   1 — ball
    #   2 — background
    # Input  (B, 3, 128, 128)
    # Output (B, 3)  — raw logits, not probabilities

    CLASSES = ["player", "ball", "background"]

    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.backbone = TennisCNNBackbone()

        # 128x128 → backbone → 256x8x8 → pool to 256x1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, len(self.CLASSES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)   # (B, 256, 8, 8)
        x = self.pool(x)       # (B, 256, 1, 1)
        return self.classifier(x)  # (B, 3)

    def predict(self, x: torch.Tensor) -> list[str]:
        # convenience method for inference — returns class name strings
        with torch.no_grad():
            indices = self.forward(x).argmax(dim=1)
        return [self.CLASSES[i] for i in indices.tolist()]


# Model 2: court keypoint detector

class TennisKeypointDetector(nn.Module):
    # Finds the 15 court landmark positions in a 640x640 image.
    # Output is normalised (x,y) coords in [0,1] — same format as CourtKeypointDataset.
    # Using AdaptiveAvgPool(4x4) instead of (1x1) because keypoints are spatial —
    # a corner in the top-left needs different features from one in the bottom-right.
    # Input  (B, 3, 640, 640)  — or any size, adaptive pool handles it
    # Output (B, 15, 2)

    NUM_KEYPOINTS = 15

    def __init__(self, dropout: float = 0.4):
        super().__init__()
        self.backbone = TennisCNNBackbone()

        # 640x640 → backbone → 256x40x40 → pool → 256x4x4 = 4096
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, self.NUM_KEYPOINTS * 2),
            nn.Sigmoid(),  # keeps output in [0,1] to match normalised targets
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)   # (B, 256, 40, 40)
        x = self.pool(x)       # (B, 256, 4, 4)
        x = self.regressor(x)  # (B, 30)
        return x.view(-1, self.NUM_KEYPOINTS, 2)  # (B, 15, 2)


# Loss for keypoint detector

def keypoint_loss(
    pred:       torch.Tensor,
    target:     torch.Tensor,
    visibility: torch.Tensor,
) -> torch.Tensor:
    # MSE loss but only over visible keypoints.
    # Dividing by visible count keeps the scale consistent even when
    # some points are occluded and masked out.
    # pred, target : (B, 15, 2)
    # visibility   : (B, 15) — 1.0 visible, 0.0 occluded

    mask          = visibility.unsqueeze(-1).expand_as(pred)
    sq_err        = ((pred - target) ** 2) * mask
    visible_count = mask.sum().clamp(min=1)
    return sq_err.sum() / visible_count
