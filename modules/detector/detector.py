"""
modules/detector/detector.py

Full-frame object detector for CourtSense.

Model:
    TennisDetector — Faster R-CNN with ResNet-50 FPN v2 backbone (COCO
                     pre-trained), fine-tuned for 2 classes: player and ball.

Tuned for broadcast footage:
  * runs at the native 720p frame size (no squashing to a square)
  * smallest anchor is 16px (default 32px) — a broadcast ball is ~5-15px
  * fine-tunes the early ResNet stages + FPN + heads, where small-object
    features live; layer4 (large, semantic) stays frozen at COCO weights

Only the fine-tuned parameters are saved (fp16, ~50MB); loading starts from
the COCO weights and overlays them.

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
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# 0 = background (reserved by Faster R-CNN), 1 = player, 2 = ball
CLASSES = ["__background__", "player", "ball"]
NUM_CLASSES = len(CLASSES)

ANCHOR_SIZES = ((16,), (32,), (64,), (128,), (256,))
ASPECT_RATIOS = ((0.5, 1.0, 2.0),) * len(ANCHOR_SIZES)
MIN_SIZE, MAX_SIZE = 720, 1280

# Parameter-name prefixes that are fine-tuned (and therefore saved).
TRAINABLE_PREFIXES = (
    "backbone.body.layer1", "backbone.body.layer2", "backbone.body.layer3",
    "backbone.fpn", "rpn", "roi_heads",
)


class TennisDetector(nn.Module):
    """
    Faster R-CNN (ResNet-50 FPN v2) fine-tuned for tennis.

    Train mode input:
        images  : list[Tensor(C, H, W)]  values in [0, 1], any size
        targets : list of dicts with:
            'boxes'  : FloatTensor(N, 4)  [x1, y1, x2, y2] absolute pixels
            'labels' : Int64Tensor(N,)    1=player, 2=ball
    Train mode output:
        dict of losses

    Eval mode output:
        list of dicts per image: 'boxes', 'labels', 'scores' (original pixel coords)
    """

    def __init__(self, pretrained_backbone: bool = True,
                 min_size: int | tuple = MIN_SIZE, max_size: int = MAX_SIZE):
        super().__init__()
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained_backbone else None
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=weights, min_size=min_size, max_size=max_size,
            trainable_backbone_layers=5,
            box_detections_per_img=50,
        )
        # Same number of anchors per location, so the pre-trained RPN head still fits.
        self.model.rpn.anchor_generator = AnchorGenerator(ANCHOR_SIZES, ASPECT_RATIOS)

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

        for name, p in self.model.named_parameters():
            p.requires_grad = name.startswith(TRAINABLE_PREFIXES)

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict] | None = None,
    ) -> dict | list[dict]:
        return self.model(images, targets)

    def trainable_state_dict(self) -> dict:
        """Fine-tuned parameters only, in fp16 (what gets saved to disk)."""
        return {k: v.detach().half().cpu() for k, v in self.model.state_dict().items()
                if k.startswith(TRAINABLE_PREFIXES)}

    def load_trainable_state_dict(self, state: dict):
        missing = [k for k in state if k not in self.model.state_dict()]
        if missing:
            raise KeyError(f"unexpected keys in detector checkpoint: {missing[:5]}")
        self.model.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)

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
            images          : Tensor(C,H,W) or (B,C,H,W) or list of tensors, values in [0,1]
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
