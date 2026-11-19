"""
Build detection labels and classifier crops from the raw datasets.

    python -m scripts.build_detection_labels            # needs keypoint_best.pt

Steps
  1. TrackNet frames (sampled): ball boxes from the per-frame ball labels.
  2. Player boxes are *pseudo-labelled*: a COCO-pretrained Faster R-CNN finds
     every person, then each person's feet are mapped onto the court with a
     homography. Per court half, the person closest to the court is the player;
     ball kids, line judges and the umpire are dropped.
       - TrackNet clips: homography from the trained court-keypoint CNN
         (median over several frames of the clip — the broadcast camera is static).
       - TennisCourtDetector frames: homography from the ground-truth keypoints.
  3. Writes Data/detection/{train,val}.json for the detector (TrackNet frames
     only: the court frames have no ball labels, so including them would teach
     the detector that visible balls are background).
  4. Extracts 64x64 player / ball / background crops for the CNN and MLP into
     Data/crops/{train,val}/{class}/, including hard negatives on court lines.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights,
                                          fasterrcnn_resnet50_fpn_v2)

from modules.cnn.train import load_keypoint_detector, predict_court_keypoints
from utils.datasets import (COURT_LINES, COURT_L, COURT_MODEL,
                            COURT_W, CROP_DIR, DATA, DET_DIR, _court_annotations,
                            ball_visible, court_image_path, crop_box_for,
                            iter_tracknet_clips, tracknet_games)

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
BALL_BOX = 14          # px, ball box side on a 1280x720 frame
PERSON_SCORE = 0.6
CROP_SIZE = 64
NET_Y = COURT_L / 2

random.seed(0)


# ── Player selection ───────────────────────────────────────────────────────────


def homography_from_points(img_pts) -> np.ndarray | None:
    """Image->court-metres homography from (up to) 14 keypoints (None = missing)."""
    src, dst = [], []
    for p, m in zip(img_pts, COURT_MODEL):
        if p is not None:
            src.append(p); dst.append(m)
    if len(src) < 4:
        return None
    H, _ = cv2.findHomography(np.float32(src), np.float32(dst), cv2.RANSAC, 0.5)
    return H


def to_court(H, x, y):
    p = H @ np.array([x, y, 1.0])
    return p[0] / p[2], p[1] / p[2]


def select_players(boxes, scores, H) -> list[list[float]]:
    """Pick at most one player per court half from person detections."""
    best = {}   # half -> (dist, -score, box)
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        cx, cy = to_court(H, (x1 + x2) / 2, y2)        # feet position in metres
        if not (-4.0 <= cx <= COURT_W + 4.0 and -7.0 <= cy <= COURT_L + 7.0):
            continue
        if abs(cy - NET_Y) < 1.5 and (cx < -0.3 or cx > COURT_W + 0.3):
            continue                                     # umpire / net judge
        dx = max(-cx, 0.0, cx - COURT_W)
        dy = max(-cy, 0.0, cy - COURT_L)
        # Players routinely stand metres behind the baseline but rarely far
        # outside the sidelines, where ball kids and line judges sit.
        dist = float(np.hypot(1.5 * dx, 0.5 * dy))
        half = "far" if cy < NET_Y else "near"
        cand = (dist, -float(score), [float(v) for v in box])
        if half not in best or cand < best[half]:
            best[half] = cand
    return [c[2] for c in best.values()]


class PersonDetector:
    def __init__(self):
        self.m = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT).to(DEVICE).eval()

    @torch.no_grad()
    def __call__(self, pil):
        out = self.m([TF.to_tensor(pil).to(DEVICE)])[0]
        keep = (out["labels"] == 1) & (out["scores"] >= PERSON_SCORE)   # COCO person
        return out["boxes"][keep].cpu().tolist(), out["scores"][keep].cpu().tolist()


# ── TrackNet ───────────────────────────────────────────────────────────────────


def clip_homography(kp_model, clip_dir: Path, rows) -> np.ndarray | None:
    names = [r["file name"] for r in rows]
    picks = [names[int(i)] for i in np.linspace(0, len(names) - 1, min(7, len(names)))]
    preds = [predict_court_keypoints(kp_model, Image.open(clip_dir / n).convert("RGB"))
             for n in picks]
    med = []
    for k in range(len(COURT_MODEL)):
        pts = [p[k] for p in preds if p[k] is not None]
        med.append(tuple(np.median(pts, axis=0)) if len(pts) >= len(preds) // 2 + 1 else None)
    return homography_from_points(med)


def build_tracknet(split: str, stride: int, persons: PersonDetector, kp_model,
                   max_clips: int = 0) -> list[dict]:
    samples = []
    for n_clip, (clip_dir, rows) in enumerate(iter_tracknet_clips(tracknet_games(split))):
        if max_clips and n_clip >= max_clips:
            break
        H = clip_homography(kp_model, clip_dir, rows)
        if H is None:
            print(f"  skip {clip_dir} (court not found)")
            continue
        for r in rows[::stride]:
            path = clip_dir / r["file name"]
            if r["visibility"] == "3":
                continue                      # occluded ball: ambiguous frame
            pil = Image.open(path).convert("RGB")
            pboxes, pscores = persons(pil)
            players = select_players(pboxes, pscores, H)
            boxes, labels = list(players), [1] * len(players)
            ball = None
            if ball_visible(r):
                bx, by = float(r["x-coordinate"]), float(r["y-coordinate"])
                h = BALL_BOX / 2
                ball = [bx - h, by - h, bx + h, by + h]
                boxes.append(ball); labels.append(2)
            samples.append({"image": str(path.relative_to(DATA)), "boxes": boxes,
                            "labels": labels, "persons": pboxes})
        print(f"  {clip_dir.parent.name}/{clip_dir.name}: {len(samples)} frames so far", flush=True)
    return samples


# ── TennisCourtDetector frames (players only, for crops) ──────────────────────


def build_court_players(split: str, persons: PersonDetector, limit: int) -> list[dict]:
    anns = _court_annotations(split)
    random.shuffle(anns)
    out = []
    for ann in anns[:limit]:
        kps = [tuple(p) if 0 <= p[0] < 1280 and 0 <= p[1] < 720 else None for p in ann["kps"]]
        H = homography_from_points([tuple(p) for p in ann["kps"]])
        if H is None:
            continue
        path = court_image_path(ann["id"])
        pil = Image.open(path).convert("RGB")
        if pil.size != (1280, 720):
            continue
        pboxes, pscores = persons(pil)
        players = select_players(pboxes, pscores, H)
        out.append({"image": str(path.relative_to(DATA)), "boxes": players,
                    "labels": [1] * len(players), "persons": pboxes, "kps": kps})
    return out


# ── Crops ──────────────────────────────────────────────────────────────────────


def _overlaps(a, b) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _random_bg_box(w, h):
    # log-uniform sizes from ball-sized to player-sized, mixed aspect ratios
    side = float(np.exp(random.uniform(np.log(10), np.log(260))))
    ar = random.choice([1.0, 1.0, 0.5, 0.4])       # square or player-shaped
    bw, bh = side * ar if ar < 1 else side, side
    x1 = random.uniform(0, max(1, w - bw)); y1 = random.uniform(0, max(1, h - bh))
    return [x1, y1, x1 + bw, y1 + bh]


def _line_boxes(kps, n):
    """Ball-sized boxes centred on court lines: the hardest negatives for a ball."""
    out = []
    lines = [(kps[a], kps[b]) for a, b in COURT_LINES if kps[a] and kps[b]]
    for _ in range(n if lines else 0):
        (x1, y1), (x2, y2) = random.choice(lines)
        t = random.random(); s = random.uniform(8, 20)
        cx, cy = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
        out.append([cx - s, cy - s, cx + s, cy + s])
    return out


def save_crops(split: str, tag: str, samples: list[dict], n_bg: int, n_line: int,
               ball_every: int = 1):
    root = CROP_DIR / split
    for c in ("player", "ball", "background"):
        (root / c).mkdir(parents=True, exist_ok=True)
    counts = {"player": 0, "ball": 0, "background": 0}
    for i, s in enumerate(samples):
        img = Image.open(DATA / s["image"]).convert("RGB")
        w, h = img.size
        occupied = []
        for box, label in zip(s["boxes"], s["labels"]):
            name = "player" if label == 1 else "ball"
            occupied.append(box)
            if name == "ball" and i % ball_every:
                continue
            crop = img.crop(crop_box_for(box, w, h)).resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR)
            crop.save(root / name / f"{tag}_{counts[name]:06d}.jpg", quality=92); counts[name] += 1
        # every detected person is excluded from background (ball kids aren't
        # "background", but they aren't the players we label either)
        occupied += s.get("persons", [])
        cands = [_random_bg_box(w, h) for _ in range(n_bg * 3)]
        if s.get("kps"):
            cands = _line_boxes(s["kps"], n_line) + cands
        kept = 0
        for box in cands:
            if kept >= n_bg + (n_line if s.get("kps") else 0):
                break
            if any(_overlaps(crop_box_for(box, w, h), o) for o in occupied):
                continue
            crop = img.crop(crop_box_for(box, w, h)).resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR)
            crop.save(root / "background" / f"{tag}_{counts['background']:06d}.jpg", quality=92)
            counts["background"] += 1; kept += 1
    print(f"  crops {split}/{tag}: {counts}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-stride", type=int, default=2)
    ap.add_argument("--val-stride", type=int, default=4)
    ap.add_argument("--court-train", type=int, default=3000)
    ap.add_argument("--court-val", type=int, default=600)
    ap.add_argument("--max-clips", type=int, default=0, help="smoke test: clips per split")
    args = ap.parse_args()

    DET_DIR.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(CROP_DIR, ignore_errors=True)     # never mix crops from older runs
    persons = PersonDetector()
    kp_model = load_keypoint_detector("best").to(DEVICE)

    for split, stride in (("val", args.val_stride), ("train", args.train_stride)):
        print(f"TrackNet {split} ...")
        tn = build_tracknet(split, stride, persons, kp_model, args.max_clips)
        with open(DET_DIR / f"{split}.json", "w") as f:
            json.dump([{k: s[k] for k in ("image", "boxes", "labels")} for s in tn], f)
        print(f"  wrote {len(tn)} frames, "
              f"{sum(2 in s['labels'] for s in tn)} with ball, "
              f"{sum(s['labels'].count(1) for s in tn)} player boxes")

        print(f"Court frames {split} ...")
        court = build_court_players(split, persons,
                                    args.court_train if split == "train" else args.court_val)
        with open(DET_DIR / f"court_players_{split}.json", "w") as f:
            json.dump(court, f)

        save_crops(split, "tracknet", tn, n_bg=1, n_line=0, ball_every=2 if split == "train" else 1)
        save_crops(split, "court", court, n_bg=2, n_line=2)
    print("Done.")
