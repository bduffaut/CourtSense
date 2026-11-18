# datasets.py
# Three PyTorch dataset classes, one per data source.
#
# PlayerDetectionDataset  feeds the CNN classifier and detector
# BallTrajectoryDataset   feeds the LSTM (sequences of ball positions)
# CourtKeypointDataset    feeds the keypoint detector (15 court landmarks)

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

_DATA = Path(__file__).parent.parent / "Data"

PLAYER_DIR = _DATA / "Tennis Player Detection.v5i.coco"
BALL_DIR = _DATA / "TennisBallDataset"
COURT_DIR = (
    _DATA / "tennis court.v8-7-court-dataset-missing-keypoint-exc-no-resize-.coco"
)


# Player Detection


class PlayerDetectionDataset(Dataset):
    # Loads player images + bounding boxes from the COCO-format dataset.

    def __init__(
        self, split: str = "train", img_size: int = 416, player_dir: Path = PLAYER_DIR
    ):
        self.img_dir = player_dir / split
        self.img_size = img_size

        with open(self.img_dir / "_annotations.coco.json") as f:
            coco = json.load(f)

        ann_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            ann_by_image[ann["image_id"]].append(ann["bbox"])

        # skip images with no annotations
        self.samples = [
            (img["file_name"], img["width"], img["height"], ann_by_image[img["id"]])
            for img in coco["images"]
            if ann_by_image[img["id"]]
        ]

        # No ImageNet normalize here — Faster R-CNN's GeneralizedRCNNTransform
        # applies it internally. Adding it here would double-normalize.
        self.transform = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name, orig_w, orig_h, bboxes = self.samples[idx]
        img = Image.open(self.img_dir / file_name).convert("RGB")
        img_tensor = self.transform(img)

        # normalise boxes to [0,1] relative to original image size
        boxes = [
            [x / orig_w, y / orig_h, w / orig_w, h / orig_h] for x, y, w, h in bboxes
        ]
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.zeros(len(boxes), dtype=torch.long),
        }
        return img_tensor, target


def detection_collate(batch):
    # custom collate because each image has a different number of boxes,
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


# Ball Trajectory


class BallTrajectoryDataset(Dataset):
    # Sliding window over ball (x,y) positions for LSTM training.
    # Input  (seq_len, 2)  normalised to [0,1]
    # Output (2,)          normalised to [0,1]

    IMG_W = 1280
    IMG_H = 720

    def __init__(self, seq_len: int = 8, ball_dir: Path = BALL_DIR):
        self.seq_len = seq_len
        self.windows = []

        for game in sorted(ball_dir.iterdir()):
            if not game.is_dir():
                continue
            for clip in sorted(game.iterdir()):
                label_file = clip / "Label.csv"
                if not label_file.exists():
                    continue
                pos = self._load_clip(label_file)
                for i in range(len(pos) - seq_len):
                    self.windows.append((pos[i : i + seq_len], pos[i + seq_len]))

    def _load_clip(self, label_file: Path) -> torch.Tensor:
        coords = []
        with open(label_file) as f:
            for row in csv.DictReader(f):
                if row["visibility"] == "0" or not row["x-coordinate"]:
                    continue
                coords.append(
                    [
                        float(row["x-coordinate"]) / self.IMG_W,
                        float(row["y-coordinate"]) / self.IMG_H,
                    ]
                )
        return torch.tensor(coords, dtype=torch.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]


# Court Keypoints


class CourtKeypointDataset(Dataset):
    # Loads court images + 15 landmark positions for the keypoint detector.
    # The 15 points mark court corners, service lines, net posts, etc.

    NUM_KP = 15

    def __init__(
        self, split: str = "train", img_size: int = 640, court_dir: Path = COURT_DIR
    ):
        self.img_dir = court_dir / split
        self.img_size = img_size

        with open(self.img_dir / "_annotations.coco.json") as f:
            coco = json.load(f)

        ann_by_image = {ann["image_id"]: ann for ann in coco["annotations"]}

        self.samples = [
            (img["file_name"], img["width"], img["height"], ann_by_image[img["id"]])
            for img in coco["images"]
            if img["id"] in ann_by_image
        ]

        self.transform = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name, orig_w, orig_h, ann = self.samples[idx]

        img = Image.open(self.img_dir / file_name).convert("RGB")
        img_tensor = self.transform(img)

        # COCO stores keypoints as a flat list
        kps = np.array(ann["keypoints"], dtype=np.float32).reshape(self.NUM_KP, 3)
        kps[:, 0] /= orig_w
        kps[:, 1] /= orig_h

        keypoints = torch.tensor(kps[:, :2], dtype=torch.float32)
        visibility = torch.tensor((kps[:, 2] > 0).astype(np.float32))

        return img_tensor, keypoints, visibility
