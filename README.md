# TennisTracker

AI-powered tennis analysis system. Five neural network architectures integrated into a single Streamlit GUI.

**Universidad San Jorge — Neural Networks / Deep Learning Group Project**

---

## What it does

Upload a tennis image or video and the system runs:

- **Faster R-CNN** — detects players and ball with bounding boxes and confidence scores
- **CNN** — classifies image crops (player / ball / background) and detects 15 court keypoint landmarks
- **MLP** — baseline fully-connected classifier on the same crops (for comparison against CNN)
- **LSTM** — predicts the next ball/player position from the trajectory history
- **VLM (Gemma 4)** — answers natural-language questions about the scene via the Chat tab

---

## Prerequisites

- **Python 3.10 or 3.11** (3.12+ not recommended — some torch builds lag behind)
- **pip**
- **Ollama** — required for the VLM chat module

---

## Step 1 — Install Ollama and the VLM model

Download and install Ollama from [https://ollama.com](https://ollama.com).

Once installed, open a terminal and pull the model:

```bash
ollama pull gemma4:e2b
```

This downloads ~2 GB. Wait for it to complete before running the app. You can verify it worked:

```bash
ollama list
# should show: gemma4:e2b
```

Ollama must be **running in the background** whenever you use the app. On macOS it starts automatically after install. On Linux run `ollama serve` in a separate terminal.

---

## Step 2 — Clone the repository

```bash
git clone <your-repo-url>
cd TennisTracker
```

---

## Step 3 — Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

This installs PyTorch, torchvision, OpenCV, Streamlit, and all other dependencies.

> **Apple Silicon (M1/M2/M3/M4):** PyTorch will automatically use MPS acceleration. No extra steps needed.
>
> **CUDA GPU:** PyTorch from `requirements.txt` may install the CPU build. For GPU support install the CUDA wheel manually from [https://pytorch.org](https://pytorch.org) before running `pip install -r requirements.txt`.

---

## Step 4 — Run the app

```bash
streamlit run gui/gui.py --server.maxUploadSize=2000
```

The `--server.maxUploadSize=2000` flag is required for video uploads (sets the limit to 2 GB). Your browser will open automatically at `http://localhost:8501`.

---

## Using the app

### Image mode
1. Select **Image** in the sidebar
2. Upload a PNG or JPG
3. Click **Run Analysis**
4. Use the checkboxes to toggle boxes, labels, confidence, and court keypoints
5. Switch to **Court View** to see player/ball positions mapped onto a 2D court
6. Switch to **Chat** and ask the VLM anything about the image

### Video mode
1. Select **Video** in the sidebar
2. Upload an MP4 / AVI / MOV (up to 2 GB)
3. Click **Process Video** — the system runs detection and tracking on every 3rd frame
4. The annotated video plays in-app with bounding boxes and trajectory trails
5. Switch to **Court View** to see the full trajectory history
6. Switch to **Chat** and ask questions — the VLM receives a representative frame from the video

### Camera mode
1. Select **Camera** in the sidebar
2. Click the shutter button to capture a still frame from your webcam
3. Click **Run Analysis** to run all models on that frame

### Training tab
Retrain any module from inside the app — configure epochs, learning rate, and batch size then click **Train**. Weights are saved automatically to each module's `weights/` folder and the models reload immediately.

> **Note:** Training requires the datasets to be present in `Data/` (see below). The pre-trained weights committed to the repo are ready to use without retraining.

---

## Datasets (only needed for retraining)

The datasets are not included in the repository. Download them and place them at these exact paths:

| Dataset | Where to get it | Path |
|---|---|---|
| Tennis Player Detection (COCO) | Roboflow — *Tennis Player Detection v5* | `Data/Tennis Player Detection.v5i.coco/` |
| Tennis Ball Trajectory | Roboflow — *TennisBallDataset* | `Data/TennisBallDataset/` |
| Tennis Court Keypoints (COCO) | Roboflow — *tennis court v8* | `Data/tennis court.v8-7-court-dataset-missing-keypoint-exc-no-resize-.coco/` |

---

## Train from the command line

Run everything from the **project root** with `-m`:

```bash
python -m modules.mlp.train --epochs 40 --lr 1e-3 --batch-size 128
python -m modules.cnn.train --model classifier --epochs 20
python -m modules.cnn.train --model keypoint --epochs 30
python -m modules.detector.train --epochs 20 --lr 0.005
python -m modules.lstm.train --epochs 50 --seq-len 8
```

---

## Project structure

```
TennisTracker/
├── gui/
│   └── gui.py                  # Streamlit application (single entry point)
├── modules/
│   ├── mlp/                    # MLP module
│   │   ├── mlp.py              # TennisMLP model definition
│   │   ├── dataset.py          # Dataset loader
│   │   ├── train.py            # Training script
│   │   └── weights/            # Saved weights
│   ├── cnn/                    # CNN module
│   │   ├── cnn.py              # TennisClassifier + TennisKeypointDetector
│   │   ├── dataset.py
│   │   ├── train.py
│   │   └── weights/
│   ├── detector/               # Object detection module
│   │   ├── detector.py         # TennisDetector (Faster R-CNN wrapper)
│   │   ├── train.py
│   │   └── weights/
│   ├── lstm/                   # Trajectory prediction module
│   │   ├── lstm.py             # TrajectoryLSTM model
│   │   ├── train.py
│   │   └── weights/
│   └── vlm/                    # Vision-Language Model module
│       └── vlm_service.py      # Gemma 4 via Ollama
├── utils/
│   └── datasets.py             # Shared PyTorch Dataset classes
├── requirements.txt
└── README.md
```

---

## Module overview

### MLP — `modules/mlp/`
Pure fully-connected network with no convolutions. Input: 32×32 grayscale crop flattened to 1024 dimensions. Four linear layers with BatchNorm, ReLU, and Dropout. Serves as a baseline to show the advantage of spatial feature extraction in the CNN.

### CNN — `modules/cnn/`
Shared `TennisCNNBackbone` (4× ConvBlock: Conv→BN→ReLU→MaxPool) with two heads:
- **TennisClassifier**: classifies 128×128 crops into player / ball / background
- **TennisKeypointDetector**: regresses 15 (x, y) court landmark coordinates from 320×320 images

### Detector — `modules/detector/`
Faster R-CNN with ResNet-50 FPN v2 backbone. Pretrained ImageNet backbone; only the detection head is fine-tuned on the tennis dataset. Detects two classes: player (label 1) and ball (label 2).

### LSTM — `modules/lstm/`
2-layer LSTM with input projection and decoder head. Takes a sliding window of 8 normalised (x, y) positions and predicts the next position. Used in the GUI to draw predicted trajectories on top of tracked detections.

### VLM — `modules/vlm/`
Gemma 4 multimodal model served locally via Ollama. Receives the current image/frame plus a structured detection context string (bounding boxes, scores, LSTM prediction). System-prompted as a tennis analyst to answer scene description and Q&A queries.

---

## Hosted demo (Docker)

The repo ships a `Dockerfile` and `docker-compose.yml` for the public image-only demo
(courtsense.benduffaut.com).

```bash
docker compose up -d --build     # serves on http://localhost:8080
```

`COURTSENSE_DEMO=1` (set by default in the image) limits the app to **Image mode** and hides
the Chat and Training tabs. Uploads are capped at 10 MB, inference runs one image at a time,
and `COURTSENSE_TORCH_THREADS` caps CPU threads. The COCO backbone weights are baked into the
image at build time, so the container starts without internet access.

The compose file builds the CUDA variant (`TORCH_VARIANT: cu132`) and reserves one NVIDIA GPU,
which requires `nvidia-container-toolkit` and an R580+ driver on the host. To run on CPU only,
set `TORCH_VARIANT: cpu` and remove the `deploy` block.
