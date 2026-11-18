"""
modules/mlp/train.py

Training loop for TennisMLP.

Usage:
    python -m modules.mlp.train
    python -m modules.mlp.train --epochs 40 --lr 5e-4 --batch-size 128
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from modules.mlp.mlp import TennisMLP
from modules.mlp.dataset import MLPCropDataset

WEIGHTS_DIR = Path(__file__).parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


def _make_balanced_sampler(dataset: MLPCropDataset) -> WeightedRandomSampler:
    counts = dataset.class_counts()
    total = sum(counts.values())
    class_weight = {
        label: total / count for label, count in counts.items() if count > 0
    }
    sample_weights = torch.tensor(
        [class_weight[label] for _, _, label in dataset.samples]
    )
    return WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )


def train(
    epochs: int = 40,
    lr: float = 1e-3,
    batch_size: int = 128,
) -> TennisMLP:

    print(f"Device: {DEVICE}")

    train_ds = MLPCropDataset(split="train", augment=True)
    valid_ds = MLPCropDataset(split="valid", augment=False)

    print("Class counts (train):", train_ds.class_counts())

    sampler = _make_balanced_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, num_workers=0
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = TennisMLP().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * features.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += features.size(0)

        train_loss /= total
        train_acc = correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for features, labels in valid_loader:
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                logits = model(features)
                loss = criterion(logits, labels)
                val_loss += loss.item() * features.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += features.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f} | "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), WEIGHTS_DIR / "mlp_best.pt")
            print(f"  saved (val_acc {val_acc:.3f})")

    torch.save(model.state_dict(), WEIGHTS_DIR / "mlp_final.pt")
    print(f"\nDone. Best val accuracy: {best_val_acc:.3f}")
    return model


def load_mlp(weights: str = "best") -> TennisMLP:
    model = TennisMLP()
    path = WEIGHTS_DIR / f"mlp_{weights}.pt"
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TennisMLP")
    parser.add_argument("--epochs",     type=int,   default=40)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int,   default=128)
    args = parser.parse_args()

    train(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
