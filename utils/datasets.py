# datasets.py
# Data sources (all broadcast footage, 1280x720):
#
#   TrackNet tennis dataset   Data/TennisBallDataset/game{1..10}/Clip*/  (+ Label.csv)
#       ~19.8k frames from 10 matches with per-frame ball positions.
#       Games 1-8 train, games 9-10 validation (split by match, never by frame,
#       so near-duplicate consecutive frames can't leak into validation).
#
#   TennisCourtDetector       Data/TennisCourtKeypoints/
#       8,841 frames from many different broadcasts, 14 court keypoints each.
#
#   Detection labels          Data/detection/{train,val}.json
#       Built by scripts/build_detection_labels.py: ball boxes from TrackNet
#       labels + player boxes pseudo-labelled with a COCO person detector and
#       filtered to the court via a homography.
#
# Run `python -m scripts.prepare_data` to download and preprocess everything.

import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

DATA = Path(__file__).parent.parent / "Data"

BALL_DIR = DATA / "TennisBallDataset"
COURT_DIR = DATA / "TennisCourtKeypoints"
DET_DIR = DATA / "detection"
CROP_DIR = DATA / "crops"

TRACKNET_TRAIN_GAMES = [f"game{i}" for i in range(1, 9)]
TRACKNET_VAL_GAMES = ["game9", "game10"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def tracknet_games(split: str) -> list[str]:
    return TRACKNET_TRAIN_GAMES if split == "train" else TRACKNET_VAL_GAMES


# ── Court geometry ─────────────────────────────────────────────────────────────
# Keypoint order used by the TennisCourtDetector dataset, as positions on a real
# court in metres (x across the court, y from the far baseline to the near one).
COURT_W = 10.97     # doubles width
COURT_L = 23.77     # baseline to baseline
_SX1, _SX2 = 1.37, 9.60          # singles sidelines
_SV_FAR, _SV_NEAR = 5.485, 18.285  # service lines
_CX = 5.485                      # centre service line
COURT_MODEL = np.array([
    [0, 0], [COURT_W, 0], [0, COURT_L], [COURT_W, COURT_L],   # 0-3 doubles corners
    [_SX1, 0], [_SX1, COURT_L], [_SX2, 0], [_SX2, COURT_L],   # 4-7 singles corners
    [_SX1, _SV_FAR], [_SX2, _SV_FAR],                          # 8-9 far service line
    [_SX1, _SV_NEAR], [_SX2, _SV_NEAR],                        # 10-11 near service line
    [_CX, _SV_FAR], [_CX, _SV_NEAR],                           # 12-13 centre T's
], dtype=np.float32)
NUM_COURT_KP = len(COURT_MODEL)
# Index permutation for a horizontal flip (left <-> right).
COURT_FLIP = [1, 0, 3, 2, 6, 7, 4, 5, 9, 8, 11, 10, 12, 13]
# Line segments between keypoints, for drawing the detected court.
COURT_LINES = [(0, 1), (2, 3), (0, 2), (1, 3), (4, 5), (6, 7),
               (8, 9), (10, 11), (12, 13)]


# ── Ball trajectories (LSTM) ───────────────────────────────────────────────────


def iter_tracknet_clips(games: list[str], ball_dir: Path = BALL_DIR):
    """Yield (clip_dir, rows) where rows are dicts from Label.csv, in frame order."""
    for game in games:
        gdir = ball_dir / game
        if not gdir.is_dir():
            continue
        for clip in sorted(gdir.iterdir(), key=lambda p: (len(p.name), p.name)):
            label_file = clip / "Label.csv"
            if not label_file.exists():
                continue
            with open(label_file) as f:
                rows = list(csv.DictReader(f))
            yield clip, rows


def ball_visible(row: dict) -> bool:
    # visibility: 0 = not in frame, 1 = visible, 2 = hard to see, 3 = occluded
    return row["visibility"] in ("1", "2") and bool(row["x-coordinate"])


class BallTrajectoryDataset(Dataset):
    """Sliding windows over *uninterrupted* runs of visible ball positions.

    Input  (seq_len, 2) normalised (x, y) in [0, 1]
    Target (2,)         normalised next position

    Windows never span a frame where the ball wasn't visible, so the model never
    learns to "teleport" across occlusions or clip boundaries.
    """

    IMG_W = 1280
    IMG_H = 720

    def __init__(self, split: str = "train", seq_len: int = 8,
                 augment: bool = False, ball_dir: Path = BALL_DIR):
        self.seq_len = seq_len
        self.augment = augment
        self.windows = []
        for _, rows in iter_tracknet_clips(tracknet_games(split), ball_dir):
            run = []
            for row in rows + [None]:
                if row is not None and ball_visible(row):
                    run.append([float(row["x-coordinate"]) / self.IMG_W,
                                float(row["y-coordinate"]) / self.IMG_H])
                    continue
                if len(run) > seq_len:
                    pos = torch.tensor(run, dtype=torch.float32)
                    for i in range(len(pos) - seq_len):
                        self.windows.append((pos[i:i + seq_len], pos[i + seq_len]))
                run = []

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        seq, target = self.windows[idx]
        if self.augment and random.random() < 0.5:   # mirror the court
            seq = seq.clone(); target = target.clone()
            seq[:, 0] = 1 - seq[:, 0]; target[0] = 1 - target[0]
        return seq, target


# ── Court keypoints (heatmap CNN) ──────────────────────────────────────────────

KP_IN_W, KP_IN_H = 640, 384      # network input (divisible by 32)
KP_STRIDE = 4                    # heatmaps at 1/4 input resolution (160x96)
KP_SIGMA = 1.5                   # gaussian radius in heatmap pixels


def _court_annotations(split: str, court_dir: Path = COURT_DIR) -> list[dict]:
    name = "data_train.json" if split == "train" else "data_val.json"
    with open(court_dir / name) as f:
        return json.load(f)


def court_image_path(img_id: str, court_dir: Path = COURT_DIR, cached: bool = True) -> Path:
    if cached:
        p = court_dir / "cache" / f"{img_id}.jpg"
        if p.exists():
            return p
    return court_dir / "images" / f"{img_id}.png"


def _gaussian_heatmaps(kps: np.ndarray, valid: np.ndarray, w: int, h: int,
                       sigma: float = KP_SIGMA) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    hm = np.zeros((len(kps), h, w), dtype=np.float32)
    for i, ((x, y), ok) in enumerate(zip(kps, valid)):
        if ok:
            # integer peak so the focal loss always has an exact positive pixel
            x, y = min(round(x), w - 1), min(round(y), h - 1)
            hm[i] = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2))
    return hm


class CourtKeypointDataset(Dataset):
    """Court frames + 14 keypoints, returned as gaussian heatmap targets.

    Returns (image, heatmaps, keypoints, valid):
        image     (3, 384, 640) ImageNet-normalised
        heatmaps  (14, 96, 160)
        keypoints (14, 2) normalised to [0, 1] of the (augmented) image
        valid     (14,)   1 if the point lies inside the image
    """

    def __init__(self, split: str = "train", augment: bool = False,
                 court_dir: Path = COURT_DIR):
        self.court_dir = court_dir
        self.samples = _court_annotations(split, court_dir)
        self.augment = augment
        self.color = T.ColorJitter(0.35, 0.35, 0.35, 0.05)
        self.norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ann = self.samples[idx]
        img = Image.open(court_image_path(ann["id"], self.court_dir)).convert("RGB")
        img.draft("RGB", (KP_IN_W, KP_IN_H))      # fast JPEG downscale on decode
        img = img.resize((KP_IN_W, KP_IN_H), Image.BILINEAR)
        # keypoints are labelled on the original 1280x720 frame
        kps = np.array(ann["kps"], dtype=np.float32) / np.array([1280, 720], np.float32)

        if self.augment:
            if random.random() < 0.5:
                img = TF.hflip(img)
                kps = kps[COURT_FLIP].copy()
                kps[:, 0] = 1 - kps[:, 0]
            # random zoom / shift (simulates different broadcast framings)
            s = random.uniform(0.85, 1.2)
            tx, ty = random.uniform(-0.08, 0.08), random.uniform(-0.08, 0.08)
            angle = random.uniform(-4, 4)
            w, h = img.size
            img = TF.affine(img, angle=angle, translate=(int(tx * w), int(ty * h)),
                            scale=s, shear=0.0)
            # apply the same transform to the points (about the image centre)
            a = math.radians(angle)   # torchvision: +angle = counter-clockwise on screen
            px = (kps[:, 0] - 0.5) * w; py = (kps[:, 1] - 0.5) * h
            rx = s * (px * math.cos(a) - py * math.sin(a)) + tx * w
            ry = s * (px * math.sin(a) + py * math.cos(a)) + ty * h
            kps = np.stack([rx / w + 0.5, ry / h + 0.5], axis=1).astype(np.float32)
            img = self.color(img)

        valid = ((kps[:, 0] >= 0) & (kps[:, 0] < 1) &
                 (kps[:, 1] >= 0) & (kps[:, 1] < 1)).astype(np.float32)
        hw, hh = KP_IN_W // KP_STRIDE, KP_IN_H // KP_STRIDE
        hm = _gaussian_heatmaps(kps * np.array([hw, hh], np.float32), valid, hw, hh)

        x = self.norm(TF.to_tensor(img))
        return (x, torch.from_numpy(hm), torch.from_numpy(kps),
                torch.from_numpy(valid))


# ── Detection (Faster R-CNN) ───────────────────────────────────────────────────


class DetectionDataset(Dataset):
    """Frames with player (label 1) and ball (label 2) boxes in absolute pixels."""

    def __init__(self, split: str = "train", augment: bool = False,
                 det_dir: Path = DET_DIR):
        with open(det_dir / f"{split}.json") as f:
            self.samples = json.load(f)
        self.root = det_dir.parent
        self.augment = augment
        self.color = T.ColorJitter(0.3, 0.3, 0.3, 0.03)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(self.root / s["image"]).convert("RGB")
        boxes = torch.tensor(s["boxes"], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(s["labels"], dtype=torch.int64)
        if self.augment:
            if random.random() < 0.5:
                img = TF.hflip(img)
                w = img.size[0]
                boxes = boxes[:, [2, 1, 0, 3]] * torch.tensor([-1, 1, -1, 1]) + \
                    torch.tensor([w, 0, w, 0])
            img = self.color(img)
        return TF.to_tensor(img), {"boxes": boxes, "labels": labels}


def detection_collate(batch):
    # images differ in size / box counts, so keep them as lists
    return [b[0] for b in batch], [b[1] for b in batch]


# ── Crops for the CNN / MLP classifiers ────────────────────────────────────────

CROP_CLASSES = ["player", "ball", "background"]


def crop_box_for(box, img_w: int, img_h: int, min_side: float = 32.0, pad: float = 0.15):
    """Square-ish context crop around a detection box.

    Used both when building the training crops and in the GUI, so the
    classifiers see detections exactly the way they were trained.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w = max(w * (1 + 2 * pad), min_side)
    h = max(h * (1 + 2 * pad), min_side)
    return (max(0, cx - w / 2), max(0, cy - h / 2),
            min(img_w, cx + w / 2), min(img_h, cy + h / 2))


class CropFolderDataset(Dataset):
    """Pre-extracted crops in Data/crops/{split}/{class}/*.jpg."""

    def __init__(self, split: str = "train", transform=None, crop_dir: Path = CROP_DIR):
        self.samples = []
        for label, name in enumerate(CROP_CLASSES):
            for p in sorted((crop_dir / split / name).glob("*.jpg")):
                self.samples.append((p, label))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return (self.transform(img) if self.transform else img), label

    def class_counts(self) -> dict:
        counts = {i: 0 for i in range(len(CROP_CLASSES))}
        for _, label in self.samples:
            counts[label] += 1
        return counts
