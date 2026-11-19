import torch
import torch.nn as nn
import torch.nn.functional as F


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
    # Classifies a detection crop (resized to 64x64) as player, ball, or background.
    # The conv layers extract features, the FC layers at the end do the classification.
    # Classes:
    # 0 : player
    # 1 : ball
    # 2 : background
    # Input  (B, 3, 64, 64)  (any size works thanks to the adaptive pool)
    # Output (B, 3)  raw logits

    CLASSES = ["player", "ball", "background"]
    INPUT_SIZE = 64

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


# ── Court keypoints ────────────────────────────────────────────────────────────


def _double_conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class TennisKeypointDetector(nn.Module):
    # Finds the 14 court landmarks (corners, service-line intersections, T's).
    #
    # An encoder-decoder CNN (U-Net style) that outputs one heatmap per
    # keypoint at 1/4 of the input resolution; decode() refines each peak to
    # sub-pixel precision. Predicting heatmaps keeps the spatial layout all the
    # way through the network, which is far more precise than regressing
    # coordinates from a globally pooled vector.
    #
    # Input  (B, 3, 384, 640)  (H, W divisible by 32)
    # Output (B, 14, 96, 160)  heatmap logits

    NUM_KEYPOINTS = 14
    STRIDE = 4

    def __init__(self, base: int = 32):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.stem = nn.Sequential(                # /2
            nn.Conv2d(3, c1, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, 3, padding=1, bias=False), nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
        )
        self.enc2 = _double_conv(c1, c2)         # /4
        self.enc3 = _double_conv(c2, c3)         # /8
        self.enc4 = _double_conv(c3, c4)         # /16
        self.bottleneck = nn.Sequential(         # /32, dilated for wide context
            nn.Conv2d(c4, c4, 3, padding=1, bias=False), nn.BatchNorm2d(c4), nn.ReLU(inplace=True),
            nn.Conv2d(c4, c4, 3, padding=2, dilation=2, bias=False), nn.BatchNorm2d(c4), nn.ReLU(inplace=True),
        )
        self.dec4 = _double_conv(c4 + c4, c3)    # /16
        self.dec3 = _double_conv(c3 + c3, c2)    # /8
        self.dec2 = _double_conv(c2 + c2, c2)    # /4
        self.head = nn.Conv2d(c2, self.NUM_KEYPOINTS, 1)
        nn.init.constant_(self.head.bias, -4.0)  # start with "no keypoint here"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.stem(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bottleneck(F.max_pool2d(e4, 2))
        d4 = self.dec4(torch.cat([F.interpolate(b, scale_factor=2), e4], 1))
        d3 = self.dec3(torch.cat([F.interpolate(d4, scale_factor=2), e3], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2), e2], 1))
        return self.head(d2)

    @staticmethod
    def decode(heatmaps: torch.Tensor):
        """Heatmap logits -> (coords (B, K, 2) normalised to [0,1], confidence (B, K)).

        Takes the peak of each heatmap and refines it with a soft-argmax over
        the 5x5 neighbourhood for sub-pixel precision.
        """
        prob = torch.sigmoid(heatmaps)
        B, K, H, W = prob.shape
        flat = prob.view(B, K, -1)
        conf, idx = flat.max(-1)
        py, px = (idx // W).float(), (idx % W).float()
        r = 2
        offs = torch.arange(-r, r + 1, device=prob.device, dtype=torch.float32)
        gy = (py[..., None] + offs).clamp(0, H - 1)            # (B, K, 5)
        gx = (px[..., None] + offs).clamp(0, W - 1)
        yy = gy[..., :, None].expand(B, K, 2 * r + 1, 2 * r + 1)
        xx = gx[..., None, :].expand(B, K, 2 * r + 1, 2 * r + 1)
        lin = (yy.long() * W + xx.long()).view(B, K, -1)
        w = torch.gather(flat, 2, lin)
        w = w / w.sum(-1, keepdim=True).clamp(min=1e-6)
        x = (w * xx.reshape(B, K, -1)).sum(-1)
        y = (w * yy.reshape(B, K, -1)).sum(-1)
        coords = torch.stack([(x + 0.5) / W, (y + 0.5) / H], -1)
        return coords, conf


def keypoint_loss(logits: torch.Tensor, target: torch.Tensor,
                  alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """Penalty-reduced focal loss on gaussian heatmaps (CenterNet).

    The heatmaps are almost entirely background, so plain MSE lets the network
    collapse to "predict zeros everywhere"; the focal loss keeps the peaks sharp.
    """
    pred = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target.ge(0.99).float()
    neg = 1 - pos
    pos_loss = torch.log(pred) * (1 - pred) ** alpha * pos
    neg_loss = torch.log(1 - pred) * pred ** alpha * (1 - target) ** beta * neg
    n_pos = pos.sum().clamp(min=1)
    return -(pos_loss.sum() + neg_loss.sum()) / n_pos
