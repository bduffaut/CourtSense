import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from modules.detector.detector import IMG_SIZE, LABEL_PLAYER, build_detector
from utils.datasets import PlayerDetectionDataset, detection_collate

WEIGHTS_DIR = Path(__file__).parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available()
                       else "cpu")


def _prepare_targets(targets: list, img_size: int, device: torch.device) -> list:
    # Dataset gives normalised xywh with label=0.
    # FasterRCNN wants absolute xyxy with label=1 (0 = background).
    prepared = []
    for t in targets:
        boxes = t['boxes'].to(device) * img_size
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1).clamp(0, img_size)
        keep       = (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & \
                     (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])
        boxes_xyxy = boxes_xyxy[keep]
        prepared.append({
            'boxes':  boxes_xyxy,
            'labels': torch.full((len(boxes_xyxy),), LABEL_PLAYER, dtype=torch.long, device=device),
        })
    return prepared


def _val_recall(model, loader, device, img_size, threshold=0.5, iou_thresh=0.5) -> float:
    model.eval()
    detected, total = 0, 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs  = [img.to(device) for img in imgs]
            preds = model(imgs)
            for pred, t in zip(preds, targets):
                gt    = t['boxes'].to(device) * img_size
                gt_xy = torch.stack([gt[:,0], gt[:,1], gt[:,0]+gt[:,2], gt[:,1]+gt[:,3]], dim=1)
                pred_boxes = pred['boxes'][pred['scores'] >= threshold]
                total += len(gt_xy)
                if len(pred_boxes) == 0:
                    continue
                for gt_box in gt_xy:
                    ix1   = torch.max(gt_box[0], pred_boxes[:, 0])
                    iy1   = torch.max(gt_box[1], pred_boxes[:, 1])
                    ix2   = torch.min(gt_box[2], pred_boxes[:, 2])
                    iy2   = torch.min(gt_box[3], pred_boxes[:, 3])
                    inter = (ix2-ix1).clamp(0) * (iy2-iy1).clamp(0)
                    area_gt   = (gt_box[2]-gt_box[0]) * (gt_box[3]-gt_box[1])
                    area_pred = (pred_boxes[:,2]-pred_boxes[:,0]) * (pred_boxes[:,3]-pred_boxes[:,1])
                    iou = inter / (area_gt + area_pred - inter).clamp(min=1e-6)
                    if iou.max() >= iou_thresh:
                        detected += 1
    return detected / max(total, 1)


def train_detector(
    epochs:     int   = 20,
    lr:         float = 5e-4,
    batch_size: int   = 2,
    img_size:   int   = IMG_SIZE,
):
    print(f"Device: {DEVICE}")

    train_ds = PlayerDetectionDataset(split='train', img_size=img_size)
    valid_ds = PlayerDetectionDataset(split='valid', img_size=img_size)
    print(f"Train: {len(train_ds)}  |  Val: {len(valid_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=detection_collate, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False,
                              collate_fn=detection_collate, num_workers=0)

    model     = build_detector().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_recall = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        model.train()
        epoch_loss, n_batches = 0.0, 0
        for imgs, targets in train_loader:
            imgs    = [img.to(DEVICE) for img in imgs]
            targets = _prepare_targets(targets, img_size, DEVICE)
            loss_dict = model(imgs, targets)
            loss      = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        epoch_loss /= max(n_batches, 1)
        scheduler.step()

        val_recall = _val_recall(model, valid_loader, DEVICE, img_size)
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"loss {epoch_loss:.4f} | "
            f"val recall {val_recall:.3f} | "
            f"{time.time() - t0:.1f}s"
        )

        if val_recall > best_recall:
            best_recall = val_recall
            torch.save(model.state_dict(), WEIGHTS_DIR / 'detector_best.pt')
            print(f"  saved (val recall {val_recall:.3f})")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), WEIGHTS_DIR / f'detector_epoch{epoch:02d}.pt')

    torch.save(model.state_dict(), WEIGHTS_DIR / 'detector_final.pt')
    print(f"\nDone. Best val recall: {best_recall:.3f}")
    return model


def load_detector():
    model = build_detector()
    model.load_state_dict(torch.load(WEIGHTS_DIR / 'detector_best.pt', map_location='cpu'))
    model.eval()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=20)
    parser.add_argument('--lr',         type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int,   default=2)
    args = parser.parse_args()

    train_detector(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
