# Generates labeled 128x128 crops for training TennisClassifier.
# Labels: 0 = player, 1 = ball, 2 = background
#
# Player crops   tight bbox crop from PlayerDetectionDataset images
# Ball crops     small patch around each visible ball from BallTrajectoryDataset frames
# Background     random crops from player images that don't overlap any bbox

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from utils.datasets import BALL_DIR, PLAYER_DIR

CROP_SIZE = 128  # everything gets resized to this
BALL_PATCH = 48  # pixels around the ball center before resizing up to 128
PLAYER_PAD = 0.15  # add 15% of bbox size as padding so we don't clip heads/feet
NEGATIVES_PER_IMG = 2  # how many background crops to try per image

PLAYER = 0
BALL = 1
BACKGROUND = 2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _overlaps(crop: tuple, bboxes: list) -> bool:
    # check if a proposed crop region intersects with any player bbox
    cx1, cy1, cx2, cy2 = crop
    for x, y, w, h in bboxes:
        if cx1 < x + w and cx2 > x and cy1 < y + h and cy2 > y:
            return True
    return False


class CropDataset(Dataset):

    def __init__(self, split: str = "train", augment: bool = False):
        self.samples = []

        self._build_player_samples(split)
        self._build_ball_samples()
        self._build_background_samples(split)

        base_transforms = [
            T.Resize((CROP_SIZE, CROP_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

        if augment:
            base_transforms = [
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            ] + base_transforms

        self.transform = T.Compose(base_transforms)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, box, label = self.samples[idx]
        img = Image.open(path).convert("RGB").crop(box)
        return self.transform(img), label

    def class_counts(self) -> dict:
        # check to see how balanced the dataset is before training
        counts = {PLAYER: 0, BALL: 0, BACKGROUND: 0}
        for _, _, label in self.samples:
            counts[label] += 1
        return counts

    def _build_player_samples(self, split: str):
        ann_path = PLAYER_DIR / split / "_annotations.coco.json"
        with open(ann_path) as f:
            coco = json.load(f)

        img_meta = {img["id"]: img for img in coco["images"]}
        ann_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            ann_by_image[ann["image_id"]].append(ann["bbox"])

        for img_id, bboxes in ann_by_image.items():
            meta = img_meta[img_id]
            path = PLAYER_DIR / split / meta["file_name"]
            img_w = meta["width"]
            img_h = meta["height"]

            for x, y, w, h in bboxes:
                pad_x = w * PLAYER_PAD
                pad_y = h * PLAYER_PAD
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(img_w, x + w + pad_x)
                y2 = min(img_h, y + h + pad_y)

                if x2 > x1 and y2 > y1:
                    self.samples.append((path, (x1, y1, x2, y2), PLAYER))

    def _build_ball_samples(self):
        for game in sorted(BALL_DIR.iterdir()):
            if not game.is_dir():
                continue
            for clip in sorted(game.iterdir()):
                label_file = clip / "Label.csv"
                if not label_file.exists():
                    continue

                with open(label_file) as f:
                    for row in csv.DictReader(f):
                        if row["visibility"] == "0" or not row["x-coordinate"]:
                            continue

                        cx = float(row["x-coordinate"])
                        cy = float(row["y-coordinate"])
                        frame_path = clip / row["file name"]

                        if not frame_path.exists():
                            continue

                        with Image.open(frame_path) as img:
                            img_w, img_h = img.size

                        half = BALL_PATCH // 2
                        x1 = max(0, cx - half)
                        y1 = max(0, cy - half)
                        x2 = min(img_w, cx + half)
                        y2 = min(img_h, cy + half)

                        if x2 > x1 and y2 > y1:
                            self.samples.append((frame_path, (x1, y1, x2, y2), BALL))

    def _build_background_samples(self, split: str):
        ann_path = PLAYER_DIR / split / "_annotations.coco.json"
        with open(ann_path) as f:
            coco = json.load(f)

        img_meta = {img["id"]: img for img in coco["images"]}
        ann_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            ann_by_image[ann["image_id"]].append(ann["bbox"])

        for img_id, bboxes in ann_by_image.items():
            meta = img_meta[img_id]
            path = PLAYER_DIR / split / meta["file_name"]
            img_w = meta["width"]
            img_h = meta["height"]

            if img_w <= CROP_SIZE or img_h <= CROP_SIZE:
                continue

            found = 0
            for _ in range(20):  # give up after 20 attempts on dense images
                if found >= NEGATIVES_PER_IMG:
                    break

                x1 = random.randint(0, img_w - CROP_SIZE)
                y1 = random.randint(0, img_h - CROP_SIZE)
                x2 = x1 + CROP_SIZE
                y2 = y1 + CROP_SIZE

                if not _overlaps((x1, y1, x2, y2), bboxes):
                    self.samples.append((path, (x1, y1, x2, y2), BACKGROUND))
                    found += 1
