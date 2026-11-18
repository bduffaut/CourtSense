"""
modules/detector/detector.py

Full-frame object detector for TennisTracker.

Model:
    TennisDetector — Faster R-CNN with ResNet-50 FPN backbone,
                     fine-tuned for 2 classes: player and ball.

Usage:
    from modules.detector.train import load_detector
    detector = load_detector('best')
    predictions = detector.detect(image_tensor)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# 0 = background (reserved by Faster R-CNN), 1 = player, 2 = ball
CLASSES = ["__background__", "player", "ball"]
NUM_CLASSES = len(CLASSES)


class TennisDetector(nn.Module):
    """
    Faster R-CNN (ResNet-50 FPN v2) fine-tuned for tennis.

    Train mode input:
        images  : list[Tensor(C, H, W)]  values in [0, 1]
        targets : list of dicts with:
            'boxes'  : FloatTensor(N, 4)  [x1, y1, x2, y2] absolute pixels
            'labels' : Int64Tensor(N,)    1=player, 2=ball
    Train mode output:
        dict of losses

    Eval mode output:
        list of dicts per image:
            'boxes'  : Tensor(N, 4)
            'labels' : Tensor(N,)
            'scores' : Tensor(N,)
    """

    def __init__(self, pretrained_backbone: bool = True):
        super().__init__()
        weights = (
            FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            if pretrained_backbone
            else None
        )
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, NUM_CLASSES
        )

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict] | None = None,
    ) -> dict | list[dict]:
        return self.model(images, targets)

    @torch.no_grad()
    def detect(
        self,
        images: list[torch.Tensor] | torch.Tensor,
        score_threshold: float = 0.5,
        device: torch.device | None = None,
    ) -> list[dict]:
        """
        Convenience inference method.

        Args:
            images          : Tensor(C,H,W) or (B,C,H,W) or list of tensors
            score_threshold : discard detections below this confidence
        Returns:
            list of dicts with 'boxes', 'labels', 'scores', 'names'
        """
        if device is None:
            device = next(self.parameters()).device

        was_training = self.training
        self.eval()

        if isinstance(images, torch.Tensor) and images.dim() == 3:
            images = [images.to(device)]
        elif isinstance(images, torch.Tensor):
            images = [img.to(device) for img in images.unbind(0)]
        else:
            images = [img.to(device) for img in images]

        raw = self.model(images)

        results = []
        for pred in raw:
            keep = pred["scores"] >= score_threshold
            results.append({
                "boxes":  pred["boxes"][keep].cpu(),
                "labels": pred["labels"][keep].cpu(),
                "scores": pred["scores"][keep].cpu(),
                "names":  [CLASSES[i] for i in pred["labels"][keep].tolist()],
            })

        if was_training:
            self.train()
        return results