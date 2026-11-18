import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    # One conv layer + batch norm + relu + maxpool.
    # Every block halves the spatial size and changes the channel count.

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


class TennisCNNBackbone(nn.Module):
    # Four conv blocks stacked, each doubles channels and halves spatial size.
    # Input  (B, 3, H, W)
    # Output (B, 256, H/16, W/16)

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TennisClassifier(nn.Module):
    # Takes a 128x128 crop and classifies it as player, ball, or background.
    # The conv layers extract features, the FC layers at the end do the classification,
    # Classes:
    # 0 : player
    # 1 : ball
    # 2 : background
    # Input  (B, 3, 128, 128)
    # Output (B, 3)  raw logits

    CLASSES = ["player", "ball", "background"]

    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.backbone = TennisCNNBackbone()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, len(self.CLASSES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x)
        return self.classifier(x)

    def predict(self, x: torch.Tensor) -> list[str]:
        # returns class name strings
        with torch.no_grad():
            indices = self.forward(x).argmax(dim=1)
        return [self.CLASSES[i] for i in indices.tolist()]


class TennisKeypointDetector(nn.Module):
    # Finds the 15 court landmark positions in a 640x640 image.
    # Input  (B, 3, 640, 640)  or any size, adaptive pool handles it
    # Output (B, 15, 2)

    NUM_KEYPOINTS = 15

    def __init__(self, dropout: float = 0.4):
        super().__init__()
        self.backbone = TennisCNNBackbone()

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
        x = self.backbone(x)
        x = self.pool(x)
        x = self.regressor(x)
        return x.view(-1, self.NUM_KEYPOINTS, 2)


# Loss for keypoint detector
def keypoint_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    visibility: torch.Tensor,
) -> torch.Tensor:

    mask = visibility.unsqueeze(-1).expand_as(pred)
    sq_err = ((pred - target) ** 2) * mask
    visible_count = mask.sum().clamp(min=1)
    return sq_err.sum() / visible_count
