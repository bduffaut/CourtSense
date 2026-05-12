import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from modules.detector.detector import (
    GRID_SIZE, IMG_SIZE, PlayerDetector, detection_loss, decode_predictions, predict, _iou
)
from utils.datasets import PlayerDetectionDataset, detection_collate

WEIGHTS_DIR = Path(__file__).parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available()
                       else "cpu")


def _build_targets(targets: list, S: int, device: torch.device) -> torch.Tensor:
    B      = len(targets)
    tensor = torch.zeros(B, S, S, 5, device=device)
    for b, t in enumerate(targets):
        for box in t['boxes'].to(device):
            x, y, w, h = box
            cx = x + w / 2
            cy = y + h / 2
            gx = min(int(cx * S), S - 1)
            gy = min(int(cy * S), S - 1)
            tensor[b, gy, gx] = torch.tensor(
                [1.0, cx * S - gx, cy * S - gy, w, h], device=device
            )
    return tensor


def _val_metrics(model, loader, device, threshold=0.5, iou_thresh=0.5) -> dict:
    model.eval()
    gt_total, tp, pred_total, n_images = 0, 0, 0, 0
    with torch.no_grad():
        for imgs, targets in loader:
            preds = model(imgs.to(device))
            for pred, t in zip(preds, targets):
                result   = decode_predictions(pred.cpu(), threshold)
                gt_boxes = t['boxes']
                gt_total    += len(gt_boxes)
                pred_total  += len(result['boxes'])
                n_images    += 1
                if len(result['boxes']) == 0 or len(gt_boxes) == 0:
                    continue
                pb      = result['boxes']
                pb_xyxy = torch.stack([pb[:,0], pb[:,1], pb[:,0]+pb[:,2], pb[:,1]+pb[:,3]], dim=1)
                for gt in gt_boxes:
                    gt_corner = torch.tensor([gt[0], gt[1], gt[0]+gt[2], gt[1]+gt[3]])
                    if _iou(gt_corner, pb_xyxy).max() >= iou_thresh:
                        tp += 1
    recall    = tp / max(gt_total, 1)
    precision = tp / max(pred_total, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-6)
    avg_preds = pred_total / max(n_images, 1)
    return {'recall': recall, 'precision': precision, 'f1': f1, 'avg_preds': avg_preds}


def train_detector(
    epochs:     int   = 50,
    lr:         float = 1e-3,
    batch_size: int   = 8,
    img_size:   int   = IMG_SIZE,
):
    print(f"Device: {DEVICE}")

    train_ds = PlayerDetectionDataset(split='train', img_size=img_size)
    valid_ds = PlayerDetectionDataset(split='valid', img_size=img_size)
    print(f"Train: {len(train_ds)}  |  Val: {len(valid_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=detection_collate, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=8, shuffle=False,
                              collate_fn=detection_collate, num_workers=0)

    model     = PlayerDetector().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1 = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        model.train()
        epoch_loss = 0.0
        epoch_obj = epoch_noobj = epoch_coord = 0.0
        n_batches = 0
        for imgs, targets in train_loader:
            imgs   = imgs.to(DEVICE)
            target = _build_targets(targets, GRID_SIZE, DEVICE)
            pred   = model(imgs)
            loss, comps = detection_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss  += loss.item()
            epoch_obj   += comps['obj']
            epoch_noobj += comps['noobj']
            epoch_coord += comps['coord']
            n_batches   += 1

        epoch_loss  /= max(n_batches, 1)
        epoch_obj   /= max(n_batches, 1)
        epoch_noobj /= max(n_batches, 1)
        epoch_coord /= max(n_batches, 1)
        lr = scheduler.get_last_lr()[0]
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            m = _val_metrics(model, valid_loader, DEVICE)
            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"loss {epoch_loss:.4f} "
                f"(obj {epoch_obj:.4f} | noobj {epoch_noobj:.4f} | coord {epoch_coord:.4f}) | "
                f"lr {lr:.2e} | {time.time() - t0:.1f}s"
            )
            print(
                f"  val → recall {m['recall']:.3f} | "
                f"precision {m['precision']:.3f} | "
                f"F1 {m['f1']:.3f} | "
                f"avg_preds {m['avg_preds']:.1f}"
            )
            if epoch >= 3 and m['f1'] > best_f1:
                best_f1 = m['f1']
                torch.save(model.state_dict(), WEIGHTS_DIR / 'detector_best.pt')
                print(f"  saved (F1 {m['f1']:.3f})")
            torch.save(model.state_dict(), WEIGHTS_DIR / f'detector_epoch{epoch:02d}.pt')
        else:
            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"loss {epoch_loss:.4f} "
                f"(obj {epoch_obj:.4f} | noobj {epoch_noobj:.4f} | coord {epoch_coord:.4f}) | "
                f"lr {lr:.2e} | {time.time() - t0:.1f}s"
            )

    torch.save(model.state_dict(), WEIGHTS_DIR / 'detector_final.pt')
    print(f"\nDone. Best val F1: {best_f1:.3f}")
    return model


def load_detector() -> PlayerDetector:
    model = PlayerDetector()
    model.load_state_dict(torch.load(WEIGHTS_DIR / 'detector_best.pt', map_location='cpu'))
    model.eval()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int,   default=8)
    args = parser.parse_args()
    train_detector(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
