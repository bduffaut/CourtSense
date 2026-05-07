import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from modules.lstm.lstm import BallTrajectoryLSTM
from utils.datasets import BallTrajectoryDataset

WEIGHTS_DIR  = Path(__file__).parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available()
                       else "cpu")

VAL_FRACTION = 0.15


def train_lstm(
    epochs:      int   = 75,
    lr:          float = 1e-3,
    batch_size:  int   = 64,
    seq_len:     int   = 8,
    hidden_size: int   = 64,
    num_layers:  int   = 2,
    dropout:     float = 0.3,
    grad_clip:   float = 1.0,
):
    print(f"Device: {DEVICE}")

    full_ds = BallTrajectoryDataset(seq_len=seq_len)
    print(f"Total windows: {len(full_ds)}")

    val_size   = int(len(full_ds) * VAL_FRACTION)
    train_size = len(full_ds) - val_size
    train_ds, valid_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {len(train_ds)}  |  Val: {len(valid_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model     = BallTrajectoryLSTM(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(DEVICE)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        model.train()
        train_loss, total = 0.0, 0
        for seq, target in train_loader:
            seq, target  = seq.to(DEVICE), target.to(DEVICE)
            target_delta = target - seq[:, -1, :]
            optimizer.zero_grad()
            pred = model(seq)
            loss = criterion(pred, target_delta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_loss += loss.item() * seq.size(0)
            total      += seq.size(0)

        train_loss /= total

        model.eval()
        val_loss, val_total = 0.0, 0
        with torch.no_grad():
            for seq, target in valid_loader:
                seq, target  = seq.to(DEVICE), target.to(DEVICE)
                target_delta = target - seq[:, -1, :]
                pred      = model(seq)
                loss      = criterion(pred, target_delta)
                val_loss += loss.item() * seq.size(0)
                val_total += seq.size(0)

        val_loss /= val_total
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train loss {train_loss:.6f} | "
            f"val loss {val_loss:.6f} | "
            f"{time.time() - t0:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), WEIGHTS_DIR / "lstm_best.pt")
            print(f"  saved (val_loss {val_loss:.6f})")

    torch.save(model.state_dict(), WEIGHTS_DIR / "lstm_final.pt")
    print(f"\nDone. Best val loss: {best_val_loss:.6f}")
    return model


def load_lstm(weights: str = "best") -> BallTrajectoryLSTM:
    model = BallTrajectoryLSTM()
    model.load_state_dict(torch.load(WEIGHTS_DIR / f"lstm_{weights}.pt", map_location="cpu"))
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,   default=75)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--seq_len",     type=int,   default=8)
    parser.add_argument("--hidden_size", type=int,   default=64)
    parser.add_argument("--num_layers",  type=int,   default=2)
    args = parser.parse_args()

    train_lstm(
        epochs      = args.epochs,
        lr          = args.lr,
        batch_size  = args.batch_size,
        seq_len     = args.seq_len,
        hidden_size = args.hidden_size,
        num_layers  = args.num_layers,
    )
