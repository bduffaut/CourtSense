# Training for TennisDetector
#
#   python -m modules.detector.train
#   python -m modules.detector.train --epochs 20 --lr 0.005

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from utils.datasets import PlayerDetectionDataset, detection_collate
from modules.detector.detector import TennisDetector

WEIGHTS_DIR = Path(__file__).parent / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


def _convert_targets(images: list[torch.Tensor], targets: list[dict]) -> list[dict]:
    """
    PlayerDetectionDataset returns normalised [x, y, w, h] boxes with label 0.
    Faster R-CNN needs absolute [x1, y1, x2, y2] boxes with label >= 1.
    """
    converted = []
    for img, t in zip(images, targets):
        _, H, W = img.shape
        b = t["boxes"]
        x1 = b[:, 0] * W
        y1 = b[:, 1] * H
        x2 = (b[:, 0] + b[:, 2]) * W
        y2 = (b[:, 1] + b[:, 3]) * H
        converted.append({
            "boxes":  torch.stack([x1, y1, x2, y2], dim=1),
            "labels": t["labels"] + 1,
        })
    return converted


def train(
    lr: float = 0.005,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    batch_size: int = 4,
    epochs: int = 20,
    val_split: float = 0.1,
) -> TennisDetector:

    print(f"Device: {DEVICE}")

    full_ds = PlayerDetectionDataset()
    n_val = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train samples: {n_train}  |  Val samples: {n_val}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=detection_collate, num_workers=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        collate_fn=detection_collate, num_workers=2,
    )

    model = TennisDetector(pretrained_backbone=True).to(DEVICE)

    # Freeze backbone — only the prediction head needs to learn.
    for p in model.model.parameters():
        p.requires_grad = False
    for p in model.model.roi_heads.box_predictor.parameters():
        p.requires_grad = True

    optimizer = torch.optim.SGD(
        model.model.roi_heads.box_predictor.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # train
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(DEVICE) for img in images]
            targets = _convert_targets(images, targets)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # validate (stay in train() to get losses; freeze BN for stable stats)
        model.train()
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(DEVICE) for img in images]
                targets = _convert_targets(images, targets)
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                val_loss += sum(loss_dict.values()).item()
        val_loss /= len(val_loader)

        scheduler.step()

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(
                model.model.roi_heads.box_predictor.state_dict(),
                WEIGHTS_DIR / "detector_best.pt",
            )

        torch.save(
            model.model.roi_heads.box_predictor.state_dict(),
            WEIGHTS_DIR / "detector_last.pt",
        )

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"{elapsed:.1f}s"
        )
        if is_best:
            print(f"  saved (val_loss {val_loss:.4f})")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    model.model.roi_heads.box_predictor.load_state_dict(
        torch.load(WEIGHTS_DIR / "detector_best.pt", map_location=DEVICE, weights_only=True)
    )
    return model


def load_detector(checkpoint: str = "best", device: str = "cpu") -> TennisDetector:
    dev = torch.device(device)
    model = TennisDetector(pretrained_backbone=True)
    path = WEIGHTS_DIR / f"detector_{checkpoint}.pt"
    model.model.roi_heads.box_predictor.load_state_dict(
        torch.load(path, map_location=dev, weights_only=True)
    )
    model.model.to(dev).eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TennisDetector")
    parser.add_argument("--lr",           type=float, default=0.005)
    parser.add_argument("--momentum",     type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--batch-size",   type=int,   default=4)
    parser.add_argument("--epochs",       type=int,   default=20)
    args = parser.parse_args()

    train(
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
