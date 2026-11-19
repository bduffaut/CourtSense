# CourtSense training report

Generated 2026-11-19 02:52 on NVIDIA GeForce RTX 4080 SUPER (torch 2.15.0+cu132, Python 3.13.1).
All metrics are on held-out validation data: TrackNet games 9-10 (matches never
seen in training) and the TennisCourtDetector validation split.

## Detector (Faster R-CNN ResNet-50 FPN v2)

935 validation frames at 720p.

| | Precision | Recall | F1 |
|---|---|---|---|
| Ball (centre within 10px, score >= 0.7) | 86.4% | 82.3% | 84.3% |
| Players (IoU >= 0.5 vs pseudo-labels) | 99.8% | 99.8% | 99.8% |

Player boxes in the validation set are pseudo-labels (COCO person detector +
court homography), so player scores measure agreement with that labeller.

## Court keypoints (heatmap CNN)

2211 validation images, 14 keypoints each.

- Mean error: **6.17px** on a 1280x720 frame
- Within 10px: **98.3%**

## Crop classifiers (player / ball / background)

| | Balanced acc. | Player | Ball | Background |
|---|---|---|---|---|
| CNN (7297 crops) | 94.7% | 99.6% | 86.1% | 98.5% |
| MLP (7297 crops) | 74.5% | 84.9% | 44.4% | 94.2% |

## Ball trajectory LSTM

3402 validation windows (8 frames -> next position, 30 fps).

- LSTM next-position error: **3.47px**
- Constant-velocity baseline: 4.52px
