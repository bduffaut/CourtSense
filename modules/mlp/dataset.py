# MLP crop dataset.
# Inherits the same player/ball/background samples as CropDataset but applies
# a flat (non-conv) transform: grayscale 32×32 → 1024-d float vector.

import torchvision.transforms as T
from modules.cnn.dataset import CropDataset
from modules.mlp.mlp import INPUT_DIM
import math

_SIDE = int(math.isqrt(INPUT_DIM))   # 32


class MLPCropDataset(CropDataset):
    """Same crops as CropDataset, returned as flat grayscale feature vectors."""

    def __init__(self, split: str = "train", augment: bool = False):
        # Let the parent build all sample lists; ignore its transform.
        super().__init__(split=split, augment=False)

        augment_tf = []
        if augment:
            augment_tf = [
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.3, contrast=0.2),
            ]

        self.transform = T.Compose([
            *augment_tf,
            T.Resize((_SIDE, _SIDE)),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),                              # (1, 32, 32)  values in [0, 1]
            T.Normalize(mean=[0.5], std=[0.5]),        # → [-1, 1]
            T.Lambda(lambda x: x.flatten()),           # (1024,)
        ])
