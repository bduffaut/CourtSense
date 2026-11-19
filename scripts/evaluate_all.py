"""
Evaluate every trained model on its validation split and write
docs/TRAINING_REPORT.md + docs/training_metrics.json.

    python -m scripts.evaluate_all

Validation data is always held out by match (TrackNet games 9-10) or by the
dataset's own split (court keypoints), so these numbers reflect footage the
models never saw during training.
"""

import json
import platform
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from modules.cnn.train import (classifier_transforms, evaluate_classifier,
                               evaluate_keypoints, load_classifier,
                               load_keypoint_detector)
from modules.detector.train import evaluate as evaluate_detector, load_detector
from modules.lstm.train import _const_velocity_px, _pixel_error, load_lstm
from modules.mlp.dataset import MLPCropDataset
from modules.mlp.train import load_mlp
from utils.datasets import (BallTrajectoryDataset, CourtKeypointDataset,
                            CropFolderDataset, DetectionDataset, detection_collate)

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")


def main():
    m = {"date": f"{datetime.now():%Y-%m-%d %H:%M}",
         "device": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else DEVICE.type,
         "torch": torch.__version__, "python": platform.python_version()}

    # ── detector ──
    det = load_detector("best", device=str(DEVICE))
    ds = DetectionDataset("val")
    dl = DataLoader(ds, batch_size=2, collate_fn=detection_collate, num_workers=2)
    sweep = {}
    for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        sweep[t] = evaluate_detector(det, dl, ball_thresh=t)
        print(f"detector ball_thresh={t}: {sweep[t]}", flush=True)
    best_t = max(sweep, key=lambda t: sweep[t]["ball_f1"])
    m["detector"] = {"val_frames": len(ds), "best_ball_thresh": best_t, **sweep[best_t],
                     "sweep": {str(k): v for k, v in sweep.items()}}
    (ROOT / "modules/detector/weights/detector_config.json").write_text(
        json.dumps({"ball_score_threshold": best_t}, indent=2))   # read by the GUI
    del det
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # ── court keypoints ──
    kp = load_keypoint_detector("best").to(DEVICE)
    kds = CourtKeypointDataset("val")
    err, pck = evaluate_keypoints(kp, DataLoader(kds, batch_size=16, num_workers=2), DEVICE)
    m["keypoints"] = {"val_images": len(kds), "mean_px_error_1280x720": err, "pck_10px": pck}
    print("keypoints:", m["keypoints"], flush=True)

    # ── classifiers ──
    cnn = load_classifier("best").to(DEVICE)
    cds = CropFolderDataset("val", classifier_transforms(False))
    acc, per = evaluate_classifier(cnn, DataLoader(cds, batch_size=256, num_workers=2), DEVICE)
    m["cnn_classifier"] = {"val_crops": len(cds), "balanced_acc": acc,
                           "recall_player": per[0], "recall_ball": per[1], "recall_bg": per[2]}
    mlp = load_mlp("best").to(DEVICE)
    mds = MLPCropDataset("val")
    acc, per = evaluate_classifier(mlp, DataLoader(mds, batch_size=256, num_workers=2), DEVICE)
    m["mlp_classifier"] = {"val_crops": len(mds), "balanced_acc": acc,
                           "recall_player": per[0], "recall_ball": per[1], "recall_bg": per[2]}
    print("cnn:", m["cnn_classifier"], "\nmlp:", m["mlp_classifier"], flush=True)

    # ── LSTM ──
    lstm = load_lstm("best", device=str(DEVICE))
    lds = BallTrajectoryDataset("val")
    m["lstm"] = {"val_windows": len(lds),
                 "next_pos_px_error": _pixel_error(lstm, DataLoader(lds, batch_size=512), DEVICE),
                 "const_velocity_px_error": _const_velocity_px(lds)}
    print("lstm:", m["lstm"], flush=True)

    out = ROOT / "docs"
    out.mkdir(exist_ok=True)
    (out / "training_metrics.json").write_text(json.dumps(m, indent=2))
    (out / "TRAINING_REPORT.md").write_text(render(m))
    print(f"\nWrote {out / 'TRAINING_REPORT.md'}")


def render(m: dict) -> str:
    d, k, c, p, l = m["detector"], m["keypoints"], m["cnn_classifier"], m["mlp_classifier"], m["lstm"]
    pct = lambda v: f"{100 * v:.1f}%"
    return f"""# CourtSense training report

Generated {m['date']} on {m['device']} (torch {m['torch']}, Python {m['python']}).
All metrics are on held-out validation data: TrackNet games 9-10 (matches never
seen in training) and the TennisCourtDetector validation split.

## Detector (Faster R-CNN ResNet-50 FPN v2)

{d['val_frames']} validation frames at 720p.

| | Precision | Recall | F1 |
|---|---|---|---|
| Ball (centre within 10px, score >= {d['best_ball_thresh']}) | {pct(d['ball_p'])} | {pct(d['ball_r'])} | {pct(d['ball_f1'])} |
| Players (IoU >= 0.5 vs pseudo-labels) | {pct(d['player_p'])} | {pct(d['player_r'])} | {pct(d['player_f1'])} |

Player boxes in the validation set are pseudo-labels (COCO person detector +
court homography), so player scores measure agreement with that labeller.

## Court keypoints (heatmap CNN)

{k['val_images']} validation images, 14 keypoints each.

- Mean error: **{k['mean_px_error_1280x720']:.2f}px** on a 1280x720 frame
- Within 10px: **{pct(k['pck_10px'])}**

## Crop classifiers (player / ball / background)

| | Balanced acc. | Player | Ball | Background |
|---|---|---|---|---|
| CNN ({c['val_crops']} crops) | {pct(c['balanced_acc'])} | {pct(c['recall_player'])} | {pct(c['recall_ball'])} | {pct(c['recall_bg'])} |
| MLP ({p['val_crops']} crops) | {pct(p['balanced_acc'])} | {pct(p['recall_player'])} | {pct(p['recall_ball'])} | {pct(p['recall_bg'])} |

## Ball trajectory LSTM

{l['val_windows']} validation windows (8 frames -> next position, 30 fps).

- LSTM next-position error: **{l['next_pos_px_error']:.2f}px**
- Constant-velocity baseline: {l['const_velocity_px_error']:.2f}px
"""


if __name__ == "__main__":
    main()
