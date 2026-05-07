import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from modules.cnn.cnn import TennisClassifier, TennisKeypointDetector, keypoint_loss
from modules.cnn.dataset import CropDataset, PLAYER, BALL, BACKGROUND
from utils.datasets import CourtKeypointDataset

WEIGHTS_DIR = Path(__file__).parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available()
                       else "cpu")


def _make_balanced_sampler(dataset: CropDataset) -> WeightedRandomSampler:
    counts = dataset.class_counts()
    total  = sum(counts.values())
    class_weight  = {label: total / count for label, count in counts.items() if count > 0}
    sample_weights = torch.tensor(
        [class_weight[label] for _, _, label in dataset.samples]
    )
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def train_classifier(
    epochs:     int   = 20,
    lr:         float = 1e-3,
    batch_size: int   = 32,
):
    print(f"Device: {DEVICE}")

    train_ds = CropDataset(split="train", augment=True)
    valid_ds = CropDataset(split="valid", augment=False)
    print("Class counts (train):", train_ds.class_counts())

    sampler      = _make_balanced_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,     num_workers=2)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model     = TennisClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

        train_loss /= total
        train_acc   = correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in valid_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits    = model(imgs)
                loss      = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total   += imgs.size(0)

        val_loss /= val_total
        val_acc   = val_correct / val_total
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f} | "
            f"{time.time() - t0:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), WEIGHTS_DIR / "classifier_best.pt")
            print(f"  saved (val_acc {val_acc:.3f})")

    torch.save(model.state_dict(), WEIGHTS_DIR / "classifier_final.pt")
    print(f"\nDone. Best val accuracy: {best_val_acc:.3f}")
    return model


def train_keypoint_detector(
    epochs:     int   = 30,
    lr:         float = 1e-3,
    batch_size: int   = 8,
):
    print(f"Device: {DEVICE}")

    train_ds = CourtKeypointDataset(split="train", img_size=320)
    valid_ds = CourtKeypointDataset(split="valid", img_size=320)
    print(f"Train samples: {len(train_ds)}  |  Valid samples: {len(valid_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model     = TennisKeypointDetector().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        model.train()
        train_loss, total = 0.0, 0
        for imgs, kps, vis in train_loader:
            imgs, kps, vis = imgs.to(DEVICE), kps.to(DEVICE), vis.to(DEVICE)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = keypoint_loss(pred, kps, vis)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            total      += imgs.size(0)

        train_loss /= total

        model.eval()
        val_loss, val_total = 0.0, 0
        with torch.no_grad():
            for imgs, kps, vis in valid_loader:
                imgs, kps, vis = imgs.to(DEVICE), kps.to(DEVICE), vis.to(DEVICE)
                pred     = model(imgs)
                loss     = keypoint_loss(pred, kps, vis)
                val_loss += loss.item() * imgs.size(0)
                val_total += imgs.size(0)

        val_loss /= val_total
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train {train_loss:.6f} (~{train_loss*640:.2f}px) | "
            f"val {val_loss:.6f} (~{val_loss*640:.2f}px) | "
            f"{time.time() - t0:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), WEIGHTS_DIR / "keypoint_best.pt")
            print(f"  saved (val_loss {val_loss:.6f})")

    torch.save(model.state_dict(), WEIGHTS_DIR / "keypoint_final.pt")
    print(f"\nDone. Best val loss: {best_val_loss:.6f} (~{best_val_loss*640:.2f}px)")
    return model


def load_classifier(weights: str = "best") -> TennisClassifier:
    model = TennisClassifier()
    model.load_state_dict(torch.load(WEIGHTS_DIR / f"classifier_{weights}.pt", map_location="cpu"))
    model.eval()
    return model


def load_keypoint_detector(weights: str = "best") -> TennisKeypointDetector:
    model = TennisKeypointDetector()
    model.load_state_dict(torch.load(WEIGHTS_DIR / f"keypoint_{weights}.pt", map_location="cpu"))
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      choices=["classifier", "keypoint", "both"], default="both")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=None)
    args = parser.parse_args()

    if args.model in ("classifier", "both"):
        train_classifier(
            epochs     = args.epochs     or 20,
            lr         = args.lr,
            batch_size = args.batch_size or 32,
        )

    if args.model in ("keypoint", "both"):
        train_keypoint_detector(
            epochs     = args.epochs     or 30,
            lr         = args.lr,
            batch_size = args.batch_size or 16,
        )
