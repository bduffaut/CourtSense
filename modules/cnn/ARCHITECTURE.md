# CNN Module — Architecture

## Overview

The CNN module is split into two models that share the same convolutional backbone but solve different problems:

| Model | Task | Input | Output |
|---|---|---|---|
| `TennisClassifier` | Classification | 128×128 image crop | 3-class probability (player / ball / background) |
| `TennisKeypointDetector` | Regression | 320×320 full court image | 15 (x,y) court landmark coordinates |

Both models are trained separately but use the same `ConvBlock` and `TennisCNNBackbone` as their foundation.

---

## The Building Block — `ConvBlock`

Every layer in the backbone is one `ConvBlock`. It always runs four operations in order:

```
Conv2d → BatchNorm2d → ReLU → MaxPool2d
```

```
Input feature map (C_in × H × W)
        ↓
  Conv2d(C_in → C_out, kernel=3×3, padding=1)
        ↓          padding=1 keeps H and W unchanged after the convolution
  BatchNorm2d(C_out)
        ↓          normalises activations across the batch — stabilises training
  ReLU
        ↓          sets negative values to 0, adds non-linearity
  MaxPool2d(2×2)
        ↓          halves H and W by keeping the max in each 2×2 region
Output (C_out × H/2 × W/2)
```

**BatchNorm** is here because without it activations can explode or vanish as gradients travel backward through many layers. It keeps everything in a stable range.

**MaxPool** reduces the spatial dimensions (saves memory and compute) and makes the features slightly translation-invariant — if a player shifts a pixel, the pooled output looks the same.

---

## The Shared Backbone — `TennisCNNBackbone`

Four `ConvBlock`s stacked in sequence. Each one doubles the number of channels and halves the spatial size.

```
Input (3 × H × W)   — 3 channels = RGB
        ↓
ConvBlock(3  → 32)   → (32  × H/2  × W/2)   — edges, colours
        ↓
ConvBlock(32 → 64)   → (64  × H/4  × W/4)   — textures, gradients
        ↓
ConvBlock(64 → 128)  → (128 × H/8  × W/8)   — parts (limbs, racket, court lines)
        ↓
ConvBlock(128 → 256) → (256 × H/16 × W/16)  — whole objects
        ↓
Output: spatial feature map  (256 × H/16 × W/16)
```

The backbone has no knowledge of the task — it only extracts visual features. The two heads attached below it decide what those features are used for.

---

## Head 1 — `TennisClassifier`

Given a 128×128 crop, this head outputs which of 3 classes it belongs to.

```
Input crop (3 × 128 × 128)
        ↓
TennisCNNBackbone
        ↓
(256 × 8 × 8)               — 128 / 16 = 8
        ↓
AdaptiveAvgPool2d(1×1)      — collapses spatial dims into a single 256-d vector
        ↓
Flatten → (256,)
        ↓
Linear(256 → 128) → ReLU → Dropout(0.5)
        ↓
Linear(128 → 3)
        ↓
Output logits (3,)          — [player, ball, background]
```

**Loss:** `CrossEntropyLoss` — softmax is applied internally so the output layer produces raw logits.

**How the training crops are generated:**
- `player` — cropped tight around each bounding box from `PlayerDetectionDataset`, resized to 128×128
- `ball` — 48×48 patch around each visible (x,y) from `BallTrajectoryDataset` frames, upscaled to 128×128
- `background` — random crop from player images in a region with no bounding box overlap

---

## Head 2 — `TennisKeypointDetector`

Given a 640×640 full court image, this head predicts the (x,y) position of 15 court landmarks.

```
Input image (3 × 640 × 640)
        ↓
TennisCNNBackbone
        ↓
(256 × 40 × 40)             — 640 / 16 = 40
        ↓
AdaptiveAvgPool2d(4×4)      — keeps a 4×4 spatial grid rather than collapsing to 1×1
        ↓
Flatten → (4096,)
        ↓
Linear(4096 → 512) → ReLU → Dropout(0.4)
        ↓
Linear(512 → 30)
        ↓
Output (30,)                — 15 keypoints × 2 values (x, y), normalised to [0,1]
```

**Loss:** `MSELoss` weighted by the visibility mask — keypoints labelled as occluded don't contribute to the gradient:
```
loss = MSE(predicted * visibility, target * visibility)
```

**Why `AdaptiveAvgPool(4×4)` here instead of `(1×1)`?**
Keypoints are spatial by nature. A corner in the top-left of the court produces different features from one in the bottom-right. Keeping a 4×4 grid preserves that structure. For the classifier it doesn't matter where in the crop the subject sits, so collapsing to (1×1) is fine there.

---

## Why the MLP Lives Inside the CNN

The spec groups "MLP / CNN Classifier" as one module because in practice the **fully connected layers at the end of a CNN are the MLP**. The conv layers do automatic feature extraction from pixels; the FC layers do the actual classification on top of those features.

```
Conv layers  =  the CNN   (extracts features from pixels automatically)
FC layers    =  the MLP   (classifies those features)
```

---

## Full Dimension Flow

```
                    TennisClassifier        TennisKeypointDetector
Input               3 × 128 × 128           3 × 320 × 320
ConvBlock 1         32 × 64 × 64            32 × 160 × 160
ConvBlock 2         64 × 32 × 32            64 × 80 × 80
ConvBlock 3         128 × 16 × 16           128 × 40 × 40
ConvBlock 4         256 × 8 × 8             256 × 20 × 20
AdaptiveAvgPool     256 × 1 × 1             256 × 4 × 4
Flatten             256                     4096
FC Layer 1          128                     512
Output              3                       30
```

---

## How This Feeds the Rest of the System

```
PlayerDetectionDataset ──┐
BallTrajectoryDataset  ──┤──► Crop Generator ──► TennisClassifier ──► player/ball/court label
CourtKeypointDataset   ──┘

CourtKeypointDataset ──────► TennisKeypointDetector ──► 15 (x,y) landmarks
                                                               ↓
                                                       Homography matrix
                                                               ↓
                                                       2D court map in GUI
```

The classifier labels feed the MLP event classifier.
The keypoint output feeds the GUI court map and gives the LSTM court-relative coordinates.

---

## Files

```
modules/cnn/
├── cnn.py           — ConvBlock, TennisCNNBackbone, TennisClassifier, TennisKeypointDetector
├── train.py         — training loops, metric logging, model checkpointing
├── dataset.py       — CropDataset (generates the 4-class patch dataset from all three sources)
└── ARCHITECTURE.md  — this file
```
