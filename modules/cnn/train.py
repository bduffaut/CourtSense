# Training for TennisClassifier and TennisKeypointDetector
#
#   python -m modules.cnn.train --model keypoint
#   python -m modules.cnn.train --model classifier
#
# Data comes from `python -m scripts.prepare_data` (see utils/datasets.py).


import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, WeightedRandomSampler

from modules.cnn.cnn import TennisClassifier, TennisKeypointDetector, keypoint_loss
from utils.datasets import (CourtKeypointDataset, CropFolderDataset, IMAGENET_MEAN,
                            IMAGENET_STD, KP_IN_H, KP_IN_W)

WEIGHTS_DIR = Path(__file__).parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)


def _workers() -> int:
    return 4 if DEVICE.type != "cpu" else 2


# Classifier training


def classifier_transforms(train: bool):
    size = TennisClassifier.INPUT_SIZE
    tf = [T.Resize((size, size)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if train:
        tf = [T.RandomHorizontalFlip(),
              T.RandomApply([T.GaussianBlur(3, (0.1, 1.2))], p=0.3),  # broadcast blur
              T.ColorJitter(0.3, 0.3, 0.25, 0.03)] + tf
    return T.Compose(tf)


def make_balanced_sampler(dataset) -> WeightedRandomSampler:
    # Gives each class equal probability regardless of how many samples it has.
    counts = dataset.class_counts()
    weight = {label: 1.0 / count for label, count in counts.items() if count > 0}
    sample_weights = torch.tensor([weight[label] for _, label in dataset.samples])
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                 replacement=True)


def train_classifier(epochs: int = 20, lr: float = 1e-3, batch_size: int = 128):
    print(f"Device: {DEVICE}")

    train_ds = CropFolderDataset("train", classifier_transforms(True))
    valid_ds = CropFolderDataset("val", classifier_transforms(False))
    print("Class counts (train):", train_ds.class_counts())
    print("Class counts (val):  ", valid_ds.class_counts())

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=make_balanced_sampler(train_ds),
                              num_workers=_workers(), persistent_workers=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              num_workers=_workers(), persistent_workers=True)

    model = TennisClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * len(train_loader))

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        val_acc, per_class = evaluate_classifier(model, valid_loader)
        print(f"Epoch {epoch:02d}/{epochs} | train loss {train_loss/total:.4f} "
              f"acc {correct/total:.3f} | val acc {val_acc:.3f} "
              f"(player {per_class[0]:.3f} ball {per_class[1]:.3f} bg {per_class[2]:.3f}) "
              f"| {time.time()-t0:.1f}s")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), WEIGHTS_DIR / "classifier_best.pt")
            print(f"  saved (val_acc {val_acc:.3f})")

    print(f"\nDone. Best val accuracy (class-balanced): {best_val_acc:.3f}")
    return model


@torch.no_grad()
def evaluate_classifier(model, loader, device=DEVICE):
    """Class-balanced accuracy (mean of per-class recall) + per-class recall."""
    model.eval()
    hits = torch.zeros(3); counts = torch.zeros(3)
    for x, labels in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        for c in range(3):
            m = labels == c
            counts[c] += m.sum(); hits[c] += (pred[m] == c).sum()
    per_class = (hits / counts.clamp(min=1)).tolist()
    return sum(per_class) / 3, per_class


# Keypoint detector training


def train_keypoint_detector(epochs: int = 40, lr: float = 2e-3, batch_size: int = 16):
    print(f"Device: {DEVICE}")

    train_ds = CourtKeypointDataset(split="train", augment=True)
    valid_ds = CourtKeypointDataset(split="val", augment=False)
    print(f"Train samples: {len(train_ds)}  |  Valid samples: {len(valid_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=_workers(), persistent_workers=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              num_workers=_workers(), persistent_workers=True)

    model = TennisKeypointDetector().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * len(train_loader), pct_start=0.1)

    best_err = float("inf")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, n = 0.0, 0
        for imgs, hms, _, _ in train_loader:
            imgs, hms = imgs.to(DEVICE), hms.to(DEVICE)
            optimizer.zero_grad()
            loss = keypoint_loss(model(imgs), hms)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item(); n += 1

        err, pck = evaluate_keypoints(model, valid_loader)
        print(f"Epoch {epoch:02d}/{epochs} | train loss {train_loss/n:.4f} | "
              f"val mean error {err:.2f}px @1280x720 | PCK@10px {pck:.3f} | "
              f"{time.time()-t0:.1f}s")
        if err < best_err:
            best_err = err
            torch.save(model.state_dict(), WEIGHTS_DIR / "keypoint_best.pt")
            print(f"  saved (val error {err:.2f}px)")

    print(f"\nDone. Best val mean error: {best_err:.2f}px (1280x720 frame)")
    return model


@torch.no_grad()
def evaluate_keypoints(model, loader, device=DEVICE):
    """Mean pixel error on a 1280x720 frame + fraction of points within 10px."""
    model.eval()
    scale = torch.tensor([1280.0, 720.0])
    errs = []
    for imgs, _, kps, valid in loader:
        coords, _ = TennisKeypointDetector.decode(model(imgs.to(device)))
        d = ((coords.cpu() - kps) * scale).norm(dim=-1)
        errs.append(d[valid.bool()])
    e = torch.cat(errs)
    return e.mean().item(), (e < 10).float().mean().item()


# Inference helpers


def load_classifier(weights: str = "best") -> TennisClassifier:
    model = TennisClassifier()
    path = WEIGHTS_DIR / f"classifier_{weights}.pt"
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def load_keypoint_detector(weights: str = "best") -> TennisKeypointDetector:
    model = TennisKeypointDetector()
    path = WEIGHTS_DIR / f"keypoint_{weights}.pt"
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def predict_court_keypoints(model: TennisKeypointDetector, pil_image,
                            min_conf: float = 0.3):
    """PIL image -> list of 14 (x, y) pixel coords, or None where not found."""
    tf = T.Compose([T.Resize((KP_IN_H, KP_IN_W)), T.ToTensor(),
                    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    dev = next(model.parameters()).device
    coords, conf = TennisKeypointDetector.decode(model(tf(pil_image).unsqueeze(0).to(dev)))
    w, h = pil_image.size
    return [(float(x) * w, float(y) * h) if c >= min_conf else None
            for (x, y), c in zip(coords[0].cpu().tolist(), conf[0].cpu().tolist())]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["classifier", "keypoint", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    if args.model in ("keypoint", "both"):
        train_keypoint_detector(epochs=args.epochs or 40, lr=args.lr or 2e-3,
                                batch_size=args.batch_size or 16)
    if args.model in ("classifier", "both"):
        train_classifier(epochs=args.epochs or 20, lr=args.lr or 1e-3,
                         batch_size=args.batch_size or 128)
