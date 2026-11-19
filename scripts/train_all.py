"""
End-to-end training pipeline (best run on a CUDA GPU).

    python -m scripts.train_all                 # full run
    python -m scripts.train_all --smoke         # tiny subset, checks everything runs

Steps
  1. prepare_data           download (~10 GB) + extract + cache into Data/
  2. court keypoints        skipped if keypoint_best.pt exists (--retrain-keypoints)
  3. detection labels       pseudo-label players, ball boxes, classifier crops
  4. detector               Faster R-CNN fine-tune
  5. CNN classifier + MLP   on the extracted crops
  6. LSTM                   skipped if lstm_best.pt exists (--retrain-lstm)
  7. report                 docs/TRAINING_REPORT.md with validation metrics

Every step can be skipped with --skip <name> (prepare, keypoints, labels,
detector, classifier, mlp, lstm, report), e.g. to resume after a crash.
"""

import argparse
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
STEPS = ["prepare", "keypoints", "labels", "detector", "classifier", "mlp", "lstm", "report"]


def run(args: list[str]):
    """Run a pipeline step as its own process (clean memory between steps)."""
    print(f"\n=== {' '.join(args)} ===", flush=True)
    t0 = time.time()
    subprocess.run([sys.executable, "-u", "-m", *args], cwd=ROOT, check=True)
    print(f"=== done in {(time.time()-t0)/60:.1f} min ===", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny subset, 1 epoch each")
    ap.add_argument("--skip", nargs="*", default=[], choices=STEPS)
    ap.add_argument("--retrain-keypoints", action="store_true")
    ap.add_argument("--retrain-lstm", action="store_true")
    ap.add_argument("--detector-epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=0, help="detector batch (0 = auto)")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    cuda = torch.cuda.is_available()
    print(f"Python {platform.python_version()} | torch {torch.__version__} | "
          f"CUDA {'yes: ' + torch.cuda.get_device_name(0) if cuda else 'NO'}")
    if not cuda and not a.smoke:
        print("WARNING: no CUDA GPU found - the detector step will be very slow.")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if cuda else 0
    bs = a.batch_size or (8 if vram >= 14 else 4 if vram >= 8 else 2)
    w = str(a.workers)
    skip = set(a.skip)
    kp_w = ROOT / "modules/cnn/weights/keypoint_best.pt"
    lstm_w = ROOT / "modules/lstm/weights/lstm_best.pt"

    if "prepare" not in skip:
        run(["scripts.prepare_data"])
    if "keypoints" not in skip and (a.retrain_keypoints or not kp_w.exists()):
        run(["modules.cnn.train", "--model", "keypoint", "--epochs", "1" if a.smoke else "25"])
    if "labels" not in skip:
        run(["scripts.build_detection_labels"] +
            (["--max-clips", "1", "--court-train", "20", "--court-val", "10"] if a.smoke else []))
    if "detector" not in skip:
        run(["modules.detector.train", "--epochs", "1" if a.smoke else str(a.detector_epochs),
             "--batch-size", str(bs), "--workers", w])
    if "classifier" not in skip:
        run(["modules.cnn.train", "--model", "classifier", "--epochs", "1" if a.smoke else "25"])
    if "mlp" not in skip:
        run(["modules.mlp.train", "--epochs", "1" if a.smoke else "40"])
    if "lstm" not in skip and (a.retrain_lstm or not lstm_w.exists()):
        run(["modules.lstm.train", "--epochs", "1" if a.smoke else "60"])
    if "report" not in skip:
        run(["scripts.evaluate_all"])
    print(f"\nAll done ({datetime.now():%Y-%m-%d %H:%M}). See docs/TRAINING_REPORT.md")


if __name__ == "__main__":
    main()
