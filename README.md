# CourtSense

AI tennis analysis system built in PyTorch. Five neural network modules feeding into a unified GUI — real-time player detection, ball trajectory prediction, court keypoint mapping, and a VLM chat interface.

**Sport:** Tennis  
**Stack:** PyTorch, torchvision, PyQt6 (GUI), HuggingFace (VLM)  
**Hardware:** Runs on MPS (Apple Silicon), CUDA, or CPU

---

## Modules

| Module | Status | What it does |
|---|---|---|
| `modules/cnn/` | ✅ Trained | Player/ball/background classifier + 15-point court keypoint detector |
| `modules/lstm/` | ✅ Trained | Ball trajectory prediction — next position from last 8 |
| `modules/detector/` | ✅ Built | Faster R-CNN player detection in full frames |
| `modules/vlm/` | 🔧 In progress | VLM scene description and Q&A |
| `gui/` | 🔧 In progress | Unified PyQt6 interface |

---

## Results

| Model | Metric |
|---|---|
| `TennisClassifier` | 99.8% val accuracy |
| `TennisKeypointDetector` | 0.04px average val error |
| `BallTrajectoryLSTM` | ~12px average error (Euclidean, 1280×720) |
| `PlayerDetector` | Train to get your own results |

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Data goes in `Data/` (not committed). Three datasets expected:
- `Tennis Player Detection.v5i.coco/` — player bounding boxes, COCO format
- `TennisBallDataset/` — per-clip CSVs with ball (x, y, visibility)
- `tennis court.v8-7-court-dataset.../` — court keypoint annotations, COCO format

---

## Training

Always run from the project root using `-m`:

```bash
# CNN classifier (player/ball/background)
python -m modules.cnn.train --model classifier

# Court keypoint detector (15 landmarks)
python -m modules.cnn.train --model keypoint

# Ball trajectory LSTM
python -m modules.lstm.train

# Player detector (Faster R-CNN) — takes ~1-1.5hr on M1
python -m modules.detector.train
```

Optional flags for all training scripts: `--epochs`, `--lr`, `--batch_size`

---

## Loading trained models

```python
from modules.cnn.train import load_classifier, load_keypoint_detector
from modules.lstm.train import load_lstm
from modules.detector.train import load_detector
from modules.detector.detector import predict as detector_predict

classifier = load_classifier('best')
keypoints  = load_keypoint_detector('best')
lstm       = load_lstm('best')
detector   = load_detector()
```

---

## Project structure

```
modules/
  cnn/        — classifier + keypoint detector (shared backbone)
  lstm/       — ball trajectory LSTM
  detector/   — Faster R-CNN player detector
  vlm/        — VLM integration (in progress)
utils/
  datasets.py — PlayerDetectionDataset, BallTrajectoryDataset, CourtKeypointDataset
gui/          — unified interface (in progress)
notebooks/    — data exploration and model sanity checks
```

All coordinates are normalised to `[0,1]` throughout. Denormalise only at render time in the GUI.
