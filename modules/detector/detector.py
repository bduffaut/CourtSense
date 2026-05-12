import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.cnn.cnn import TennisCNNBackbone

GRID_SIZE = 26   # backbone outputs 26×26 for a 416×416 input (416/16 = 26)
IMG_SIZE  = 416


class PlayerDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TennisCNNBackbone()
        self.head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 5, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)         # (B, 256, 26, 26)
        out      = self.head(features)      # (B, 5, 26, 26)
        return out.permute(0, 2, 3, 1)      # (B, 26, 26, 5)


def _focal_loss(pred: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    p_t = pred * target + (1 - pred) * (1 - target)
    return ((1 - p_t) ** gamma * bce).mean()


def detection_loss(
    pred:         torch.Tensor,
    target:       torch.Tensor,
    lambda_coord: float = 5.0,
    lambda_noobj: float = 3.0,
) -> tuple:
    obj_mask   = target[..., 0] == 1
    noobj_mask = ~obj_mask

    zero = pred.sum() * 0   # keeps graph alive when a mask is empty

    loss_obj   = _focal_loss(pred[..., 0][obj_mask],   target[..., 0][obj_mask])   if obj_mask.any()   else zero
    loss_noobj = _focal_loss(pred[..., 0][noobj_mask], target[..., 0][noobj_mask]) if noobj_mask.any() else zero
    loss_xy    = F.mse_loss(pred[..., 1:3][obj_mask],  target[..., 1:3][obj_mask]) if obj_mask.any()   else zero
    loss_wh    = F.mse_loss(pred[..., 3:5][obj_mask],  target[..., 3:5][obj_mask]) if obj_mask.any()   else zero

    total = lambda_coord * (loss_xy + loss_wh) + loss_obj + lambda_noobj * loss_noobj
    return total, {
        'obj':   loss_obj.item(),
        'noobj': loss_noobj.item(),
        'coord': (loss_xy + loss_wh).item(),
    }


def _iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    ax1, ay1 = box[0], box[1]
    ax2, ay2 = box[0] + box[2], box[1] + box[3]
    bx1, by1 = boxes[:, 0], boxes[:, 1]
    bx2, by2 = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
    inter = (torch.min(ax2, bx2) - torch.max(ax1, bx1)).clamp(0) * \
            (torch.min(ay2, by2) - torch.max(ay1, by1)).clamp(0)
    return inter / (box[2]*box[3] + boxes[:,2]*boxes[:,3] - inter + 1e-6)


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float = 0.5) -> torch.Tensor:
    order = scores.argsort(descending=True)
    keep  = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        order = order[1:][_iou(boxes[i], boxes[order[1:]]) < iou_thresh]
    return torch.tensor(keep, dtype=torch.long)


def decode_predictions(
    pred:      torch.Tensor,
    threshold: float = 0.5,
    nms_iou:   float = 0.5,
    max_det:   int   = 0,      # 0 = no limit; set 2 in GUI for tennis (always 2 players)
) -> dict:
    S      = pred.shape[0]
    gy, gx = (pred[..., 0] >= threshold).nonzero(as_tuple=True)

    if gy.numel() == 0:
        return {'boxes': torch.zeros(0, 4), 'scores': torch.zeros(0)}

    tx, ty = pred[gy, gx, 1], pred[gy, gx, 2]
    tw, th = pred[gy, gx, 3], pred[gy, gx, 4]
    scores = pred[gy, gx, 0]
    cx     = (gx.float() + tx) / S
    cy     = (gy.float() + ty) / S
    boxes  = torch.stack([cx - tw / 2, cy - th / 2, tw, th], dim=1)

    keep = _nms(boxes, scores, nms_iou)
    if max_det > 0:
        keep = keep[:max_det]
    return {'boxes': boxes[keep], 'scores': scores[keep]}


def predict(
    model:     PlayerDetector,
    image:     torch.Tensor,
    device:    torch.device,
    threshold: float = 0.5,
    nms_iou:   float = 0.5,
    max_det:   int   = 0,
) -> dict:
    model.eval()
    with torch.no_grad():
        pred = model(image.unsqueeze(0).to(device))[0].cpu()
    return decode_predictions(pred, threshold, nms_iou, max_det)
