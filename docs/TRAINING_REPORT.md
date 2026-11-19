# CourtSense training report

Generated 2026-11-19 00:46 on mps (torch 2.15.0, Python 3.11.17).
All metrics are on held-out validation data: TrackNet games 9-10 (matches never
seen in training) and the TennisCourtDetector validation split.

## Detector (Faster R-CNN ResNet-50 FPN v2)

21 validation frames at 720p.

| | Precision | Recall | F1 |
|---|---|---|---|
| Ball (centre within 10px, score >= 0.1) | 0.0% | 0.0% | 0.0% |
| Players (IoU >= 0.5 vs pseudo-labels) | 0.0% | 0.0% | 0.0% |

Player boxes in the validation set are pseudo-labels (COCO person detector +
court homography), so player scores measure agreement with that labeller.

## Court keypoints (heatmap CNN)

2211 validation images, 14 keypoints each.

- Mean error: **6.17px** on a 1280x720 frame
- Within 10px: **98.3%**

## Crop classifiers (player / ball / background)

| | Balanced acc. | Player | Ball | Background |
|---|---|---|---|---|
| CNN (143 crops) | 36.0% | 8.1% | 100.0% | 0.0% |
| MLP (143 crops) | 47.6% | 67.7% | 0.0% | 75.0% |

## Ball trajectory LSTM

3402 validation windows (8 frames -> next position, 30 fps).

- LSTM next-position error: **3.47px**
- Constant-velocity baseline: 4.52px
