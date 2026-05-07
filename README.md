# TennisTracker

AI tennis analysis system built in PyTorch. Five neural network modules + unified GUI.
Sport: Tennis. Datasets are in `Data/` (not committed to git).

---

## What's been built

### `utils/datasets.py`

Three PyTorch Dataset classes. Everything else imports from here.

- `PlayerDetectionDataset` — player images + bounding boxes (COCO format)
- `BallTrajectoryDataset` — sliding window of ball (x,y) positions for LSTM
- `CourtKeypointDataset` — court images + 15 landmark positions

### `modules/cnn/`

CNN module — fully done and trained.

| File              | What it does                                                                                  |
| ----------------- | --------------------------------------------------------------------------------------------- |
| `cnn.py`          | `TennisClassifier` (player/ball/background) and `TennisKeypointDetector` (15 court landmarks) |
| `dataset.py`      | Generates 128x128 labeled crops from the raw datasets for classifier training                 |
| `train.py`        | Training loops for both models. Run with `python -m modules.cnn.train --model classifier`     |
| `ARCHITECTURE.md` | Full architecture breakdown with dimension flows                                              |

**Trained weights** are in `modules/cnn/weights/`:

- `classifier_best.pt` — 99.8% val accuracy (player/ball/background)
- `keypoint_best.pt` — 0.04px val error (15 court landmarks)

Load them with:

```python
from modules.cnn.train import load_classifier, load_keypoint_detector
classifier = load_classifier('best')        # TennisClassifier, ready for inference
keypoints  = load_keypoint_detector('best') # TennisKeypointDetector, ready for inference
```

### `notebooks/`

- `01_dataloaders.ipynb` — explores all three datasets, visualises samples
- `02_cnn.ipynb` — sanity checks for the CNN module, test training runs to make sure it works before long training session

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run everything from the project root using `-m`:

```bash
python -m modules.cnn.train --model classifier
python -m modules.cnn.train --model keypoint
```

---

## What's left

**Person 2 — `modules/lstm/`**
LSTM that takes sequences of (x,y) player/ball detections and predicts the next position.
`BallTrajectoryDataset` in `utils/datasets.py` is ready to use — it already outputs `(seq_len, 2)` sequences.

**Person 2 — `modules/detector/`**
Object detection model that finds players and ball in a full frame with bounding boxes.
`PlayerDetectionDataset` in `utils/datasets.py` is ready to use.

**Person 3 — `modules/vlm/`**
VLM/Transformer module — generates natural language descriptions of what's happening on court.
Recommended: LLaVA, MiniCPM-V, or Qwen-VL via HuggingFace.

**Person 3 — `gui/`**
Unified GUI that connects all modules. Must support image/video/camera input,
live detection display, trajectory overlay, and a chat panel for the VLM.
