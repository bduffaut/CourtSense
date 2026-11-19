"""
modules/lstm/train.py

Training loop for TrajectoryLSTM.

Usage:
    python -m modules.lstm.train
    python -m modules.lstm.train --epochs 50 --seq-len 20

Load trained weights:
    from modules.lstm.train import load_lstm
    model = load_lstm('best')
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.datasets import BallTrajectoryDataset
from modules.lstm.lstm import TrajectoryLSTM

WEIGHTS_DIR = Path(__file__).parent / "weights"


def train(
    seq_len: int = 8,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 1e-3,
    batch_size: int = 64,
    epochs: int = 50,
    val_split: float = 0.1,
    device: str | None = None,
) -> TrajectoryLSTM:

    dev = torch.device(
        device or (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    )
    print(f"Training on {dev}")

    # -- Data ------------------------------------------------------------------
    # Split by match (games 1-8 / 9-10): overlapping windows from the same rally
    # would otherwise appear in both sets and make validation meaningless.
    train_ds = BallTrajectoryDataset("train", seq_len=seq_len, augment=True)
    val_ds = BallTrajectoryDataset("val", seq_len=seq_len)
    print(f"Windows: train {len(train_ds)} | val {len(val_ds)}")
    print(f"Baseline (constant velocity) val error: {_const_velocity_px(val_ds):.2f}px")
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0,
    )

    # -- Model -----------------------------------------------------------------
    model = TrajectoryLSTM(
        input_size=2,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(dev)

    # Huber in pixel-ish units: robust to the occasional bounce / hit, where the
    # next position is genuinely unpredictable from the past
    criterion = nn.SmoothL1Loss(beta=0.005)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # -- Train --
        model.train()
        train_loss = 0.0
        for sequences, targets in train_loader:
            # sequences : (batch, seq_len, 2)
            # targets   : (batch, 2)  <- BallTrajectoryDataset returns single point
            sequences = sequences.to(dev)
            targets = targets.unsqueeze(1).to(dev)   # (batch, 2) -> (batch, 1, 2)

            optimizer.zero_grad()
            preds = model(sequences)                 # (batch, 1, 2)
            loss = criterion(preds, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * sequences.size(0)

        train_loss /= len(train_ds)

        # -- Validate --
        model.eval()
        val_loss = _pixel_error(model, val_loader, dev)   # mean px error @1280x720
        scheduler.step()

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), WEIGHTS_DIR / "lstm_best.pt")
        torch.save(model.state_dict(), WEIGHTS_DIR / "lstm_last.pt")

        print(
            f"Epoch {epoch:03d}/{epochs}  "
            f"train={train_loss:.6f}  val error={val_loss:.2f}px"
            + (" * best" if is_best else "")
        )

    print(f"\nBest val error: {best_val_loss:.2f}px "
          f"(constant-velocity baseline {_const_velocity_px(val_ds):.2f}px)")
    model.load_state_dict(
        torch.load(WEIGHTS_DIR / "lstm_best.pt", map_location=dev, weights_only=True)
    )
    return model


_SCALE = torch.tensor([BallTrajectoryDataset.IMG_W, BallTrajectoryDataset.IMG_H],
                      dtype=torch.float32)


@torch.no_grad()
def _pixel_error(model, loader, dev) -> float:
    total, n = 0.0, 0
    for seq, tgt in loader:
        pred = model(seq.to(dev)).squeeze(1).cpu()
        total += ((pred - tgt) * _SCALE).norm(dim=-1).sum().item()
        n += len(tgt)
    return total / max(n, 1)


def _const_velocity_px(ds) -> float:
    """Error of the trivial 'keep moving at the same speed' predictor."""
    seqs = torch.stack([w[0] for w in ds.windows]); tgts = torch.stack([w[1] for w in ds.windows])
    pred = seqs[:, -1] + (seqs[:, -1] - seqs[:, -2])
    return ((pred - tgts) * _SCALE).norm(dim=-1).mean().item()


def load_lstm(
    checkpoint: str = "best",
    seq_len: int = 8,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    device: str | None = None,
) -> TrajectoryLSTM:
    """
    Load a trained TrajectoryLSTM from weights/.

    Args:
        checkpoint : 'best' or 'last'
    Returns:
        model in eval() mode
    """
    dev = torch.device(
        device or (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    )
    model = TrajectoryLSTM(
        input_size=2,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    path = WEIGHTS_DIR / f"lstm_{checkpoint}.pt"
    model.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
    model.to(dev).eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TrajectoryLSTM")
    parser.add_argument("--seq-len",    type=int,   default=8)
    parser.add_argument("--hidden",     type=int,   default=128)
    parser.add_argument("--layers",     type=int,   default=2)
    parser.add_argument("--dropout",    type=float, default=0.2)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--val-split",  type=float, default=0.1)
    parser.add_argument("--device",     type=str,   default=None)
    args = parser.parse_args()

    train(
        seq_len=args.seq_len,
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_split=args.val_split,
        device=args.device,
    )