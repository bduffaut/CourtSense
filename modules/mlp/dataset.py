# MLP crop dataset.
# Same player/ball/background crops as the CNN (Data/crops/), but flattened:
# RGB 32×32 → 3072-d float vector, so the MLP sees raw pixels with no
# spatial structure (the point of the CNN-vs-MLP comparison).

import torchvision.transforms as T

from modules.mlp.mlp import SIDE
from utils.datasets import CropFolderDataset


def _flatten(x):
    return x.flatten()


def mlp_transforms(train: bool):
    aug = [T.RandomHorizontalFlip(), T.ColorJitter(0.3, 0.3, 0.2, 0.02)] if train else []
    return T.Compose([
        *aug,
        T.Resize((SIDE, SIDE)),
        T.ToTensor(),                              # (3, 32, 32) in [0, 1]
        T.Normalize([0.5] * 3, [0.5] * 3),         # → [-1, 1]
        T.Lambda(_flatten),                        # (3072,)
    ])


class MLPCropDataset(CropFolderDataset):
    """Same crops as the CNN classifier, returned as flat RGB feature vectors."""

    def __init__(self, split: str = "train", augment: bool = False):
        super().__init__(split, mlp_transforms(augment))
