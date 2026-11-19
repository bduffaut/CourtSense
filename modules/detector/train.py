# Training for TennisDetector
#
#   python -m scripts.build_detection_labels     # once, builds Data/detection/
#   python -m modules.detector.train
#   python -m modules.detector.train --epochs 8 --batch-size 4
#
# Uses CUDA mixed precision when available (e.g. the RTX 2060 homeserver).

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from utils.datasets import DetectionDataset, detection_collate
from modules.detector.detector import TennisDetector

WEIGHTS_DIR = Path(__file__).parent / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

BALL_HIT_PX = 10.0   # a ball detection counts if its centre is within 10px


def train(
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    batch_size: int = 4,
    epochs: int = 8,
    workers: int = 4,
    max_val: int = 600,
) -> TennisDetector:

    print(f"Device: {DEVICE}")
    train_ds = DetectionDataset("train", augment=True)
    val_ds = DetectionDataset("val")
    if max_val and len(val_ds) > max_val:   # evenly spaced subset keeps eval quick
        step = len(val_ds) / max_val
        val_ds.samples = [val_ds.samples[int(i * step)] for i in range(max_val)]
    print(f"Train images: {len(train_ds)}  |  Val images: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=detection_collate, num_workers=workers,
                              persistent_workers=workers > 0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False,
                            collate_fn=detection_collate, num_workers=workers)

    # multi-scale training around the 720p inference size
    model = TennisDetector(pretrained_backbone=True, min_size=(640, 672, 720, 768, 800)).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in params)/1e6:.1f}M")

    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    total = epochs * len(train_loader)
    warmup = min(500, len(train_loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda it: min(1.0, (it + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * it / total)))
    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = -1.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running, n = 0.0, 0
        for i, (images, targets) in enumerate(train_loader):
            images = [img.to(DEVICE, non_blocking=True) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            with torch.autocast(DEVICE.type, enabled=use_amp):
                loss = sum(model(images, targets).values())
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(optimizer); scaler.update()
            scheduler.step()
            running += loss.item(); n += 1
            if i % 100 == 0:
                print(f"  epoch {epoch} iter {i}/{len(train_loader)} loss {running/n:.4f} "
                      f"lr {scheduler.get_last_lr()[0]:.5f} ({time.time()-t0:.0f}s)", flush=True)

        model.model.transform.min_size = (720,)
        m = evaluate(model, val_loader)
        model.model.transform.min_size = (640, 672, 720, 768, 800)
        score = m["ball_f1"] + m["player_f1"]
        print(f"Epoch {epoch:02d}/{epochs} | train loss {running/n:.4f} | "
              f"ball P {m['ball_p']:.3f} R {m['ball_r']:.3f} F1 {m['ball_f1']:.3f} | "
              f"player P {m['player_p']:.3f} R {m['player_r']:.3f} F1 {m['player_f1']:.3f} | "
              f"{time.time()-t0:.0f}s", flush=True)
        torch.save(model.trainable_state_dict(), WEIGHTS_DIR / "detector_last.pt")
        if score > best:
            best = score
            torch.save(model.trainable_state_dict(), WEIGHTS_DIR / "detector_best.pt")
            print("  saved best", flush=True)

    return model


@torch.no_grad()
def evaluate(model, loader, ball_thresh: float = 0.3, player_thresh: float = 0.5) -> dict:
    """Ball: top-scoring ball per frame, hit if centre within BALL_HIT_PX of the label.
    Players: IoU >= 0.5 matching against the (pseudo-)labelled player boxes."""
    model.eval()
    dev = next(model.parameters()).device
    b_tp = b_fp = b_fn = 0
    p_tp = p_fp = p_fn = 0
    for images, targets in loader:
        outs = model([img.to(dev) for img in images])
        for out, tgt in zip(outs, targets):
            out = {k: v.cpu() for k, v in out.items()}
            # ball
            gt_b = tgt["boxes"][tgt["labels"] == 2]
            sel = (out["labels"] == 2) & (out["scores"] >= ball_thresh)
            pb = out["boxes"][sel]
            if len(pb):
                c = (pb[0, :2] + pb[0, 2:]) / 2
                if len(gt_b) and ((gt_b[:, :2] + gt_b[:, 2:]) / 2 - c).norm(dim=-1).min() <= BALL_HIT_PX:
                    b_tp += 1
                else:
                    b_fp += 1
                    b_fn += int(len(gt_b) > 0)
            else:
                b_fn += int(len(gt_b) > 0)
            # players
            gt_p = tgt["boxes"][tgt["labels"] == 1]
            sel = (out["labels"] == 1) & (out["scores"] >= player_thresh)
            pp = out["boxes"][sel][:2]
            if len(pp) and len(gt_p):
                iou = box_iou(pp, gt_p)
                matched = set()
                for r in range(len(pp)):
                    j = int(iou[r].argmax())
                    if iou[r, j] >= 0.5 and j not in matched:
                        matched.add(j); p_tp += 1
                    else:
                        p_fp += 1
                p_fn += len(gt_p) - len(matched)
            else:
                p_fp += len(pp); p_fn += len(gt_p)

    def prf(tp, fp, fn):
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        return p, r, 2 * p * r / max(p + r, 1e-9)
    bp, br, bf = prf(b_tp, b_fp, b_fn)
    pp_, pr, pf = prf(p_tp, p_fp, p_fn)
    return {"ball_p": bp, "ball_r": br, "ball_f1": bf,
            "player_p": pp_, "player_r": pr, "player_f1": pf}


def load_detector(checkpoint: str = "best", device: str = "cpu") -> TennisDetector:
    dev = torch.device(device)
    model = TennisDetector(pretrained_backbone=True)
    path = WEIGHTS_DIR / f"detector_{checkpoint}.pt"
    model.load_trainable_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.model.to(dev).eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TennisDetector")
    parser.add_argument("--lr",           type=float, default=0.01)
    parser.add_argument("--momentum",     type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size",   type=int,   default=4)
    parser.add_argument("--epochs",       type=int,   default=8)
    parser.add_argument("--workers",      type=int,   default=4)
    args = parser.parse_args()

    train(
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        workers=args.workers,
    )
