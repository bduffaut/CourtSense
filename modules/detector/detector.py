from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from modules.cnn.cnn import TennisCNNBackbone

IMG_SIZE     = 416
LABEL_PLAYER = 1   # 0 is reserved for background in FasterRCNN


class _BackboneWrapper(nn.Module):
    # FasterRCNN expects the backbone to return a named dict, not a raw tensor
    out_channels = 256

    def __init__(self):
        super().__init__()
        self.body = TennisCNNBackbone()

    def forward(self, x: torch.Tensor) -> dict:
        return OrderedDict([('0', self.body(x))])


def build_detector() -> FasterRCNN:
    anchor_gen = AnchorGenerator(
        sizes=((32, 64, 128, 256),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    )
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2,
    )
    return FasterRCNN(
        backbone             = _BackboneWrapper(),
        num_classes          = 2,
        rpn_anchor_generator = anchor_gen,
        box_roi_pool         = roi_pooler,
        min_size             = IMG_SIZE,
        max_size             = IMG_SIZE,
    )


def predict(
    model:     FasterRCNN,
    image:     torch.Tensor,
    device:    torch.device,
    threshold: float = 0.5,
) -> dict:
    # Returns normalised [x, y, w, h] boxes — consistent with the rest of the system.
    model.eval()
    with torch.no_grad():
        output = model([image.to(device)])[0]

    mask   = output['scores'] >= threshold
    boxes  = output['boxes'][mask]
    scores = output['scores'][mask]

    _, H, W = image.shape
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    norm_boxes = torch.stack([x1/W, y1/H, (x2-x1)/W, (y2-y1)/H], dim=1)

    return {'boxes': norm_boxes, 'scores': scores}
