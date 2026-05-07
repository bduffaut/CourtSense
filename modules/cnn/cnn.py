import torch
import torch.nn as nn


class ConvBlock(nn.Module):
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
    # (B, 3, H, W) -> (B, 256, H/16, W/16)
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(3,   32),
            ConvBlock(32,  64),
            ConvBlock(64,  128),
            ConvBlock(128, 256),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TennisClassifier(nn.Module):
    # 128x128 crop -> player / ball / background
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
        with torch.no_grad():
            indices = self.forward(x).argmax(dim=1)
        return [self.CLASSES[i] for i in indices.tolist()]


class TennisKeypointDetector(nn.Module):
    # Full court image -> 15 landmark (x,y) coords, normalised to [0,1]
    NUM_KEYPOINTS = 15

    def __init__(self, dropout: float = 0.4):
        super().__init__()
        self.backbone = TennisCNNBackbone()
        # 4x4 keeps spatial structure — a corner top-left has different features than bottom-right
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, self.NUM_KEYPOINTS * 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x)
        x = self.regressor(x)
        return x.view(-1, self.NUM_KEYPOINTS, 2)


def keypoint_loss(
    pred:       torch.Tensor,
    target:     torch.Tensor,
    visibility: torch.Tensor,
) -> torch.Tensor:
    # only penalise visible keypoints
    mask   = visibility.unsqueeze(-1).expand_as(pred)
    sq_err = ((pred - target) ** 2) * mask
    return sq_err.sum() / mask.sum().clamp(min=1)
