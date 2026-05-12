import sys
import subprocess
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from PyQt6.QtCore import QThread, pyqtSignal

DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available()  else
    "cuda" if torch.cuda.is_available()            else
    "cpu"
)

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def _preprocess(pil_img: Image.Image, size: int) -> torch.Tensor:
    img = pil_img.convert("RGB").resize((size, size), Image.BILINEAR)
    t = TF.to_tensor(img)
    return TF.normalize(t, _MEAN, _STD)


class ModelLoader(QThread):
    """Loads all trained models in a background thread; emits ready when done."""
    progress = pyqtSignal(str)
    ready    = pyqtSignal(dict)

    def run(self):
        models = {}

        try:
            self.progress.emit("Loading classifier…")
            from modules.cnn.train import load_classifier
            models['classifier'] = load_classifier('best').to(DEVICE).eval()
        except Exception as e:
            self.progress.emit(f"Classifier unavailable ({e})")

        try:
            self.progress.emit("Loading keypoint detector…")
            from modules.cnn.train import load_keypoint_detector
            models['keypoint'] = load_keypoint_detector('best').to(DEVICE).eval()
        except Exception as e:
            self.progress.emit(f"Keypoint detector unavailable ({e})")

        try:
            self.progress.emit("Loading LSTM…")
            from modules.lstm.train import load_lstm
            models['lstm'] = load_lstm('best').to(DEVICE).eval()
        except Exception as e:
            self.progress.emit(f"LSTM unavailable ({e})")

        try:
            self.progress.emit("Loading player detector…")
            from modules.detector.train import load_detector
            models['detector'] = load_detector().to(DEVICE).eval()
        except Exception as e:
            self.progress.emit(f"Detector unavailable ({e})")

        try:
            self.progress.emit("Loading VLM (first run downloads ~4 GB)…")
            from modules.vlm.vlm import load_vlm
            models['vlm_model'], models['vlm_processor'] = load_vlm()
        except Exception as e:
            self.progress.emit(f"VLM unavailable ({e})")

        self.ready.emit(models)


class InferenceWorker(QThread):
    """Runs player detection + court keypoint detection on one frame."""
    result = pyqtSignal(dict)

    def __init__(self, models: dict, pil_image: Image.Image):
        super().__init__()
        self._models = models
        self._pil    = pil_image

    def run(self):
        torch.set_grad_enabled(False)
        out = {}
        try:
            det = self._models.get('detector')
            if det:
                from modules.detector.detector import predict as det_predict
                t = _preprocess(self._pil, 416)
                r = det_predict(det, t, DEVICE, max_det=2)
                out['boxes']  = r['boxes'].cpu()
                out['scores'] = r['scores'].cpu()

            kp = self._models.get('keypoint')
            if kp:
                t = _preprocess(self._pil, 320).unsqueeze(0).to(DEVICE)
                out['keypoints'] = kp(t)[0].cpu()

        except Exception as e:
            out['error'] = str(e)

        self.result.emit(out)


class VLMWorker(QThread):
    """Runs a VLM query or describe_scene call in a background thread."""
    response = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(self, vlm_model, vlm_processor, pil_image: Image.Image,
                 prompt: str, context: dict | None = None):
        super().__init__()
        self._model     = vlm_model
        self._processor = vlm_processor
        self._image     = pil_image
        self._prompt    = prompt
        self._context   = context

    def run(self):
        try:
            from modules.vlm.vlm import query, describe_scene
            if self._context is not None:
                text = describe_scene(self._model, self._processor,
                                      self._image, self._context)
            else:
                text = query(self._model, self._processor,
                             self._image, self._prompt)
            self.response.emit(text)
        except Exception as e:
            self.error.emit(str(e))


class TrainingWorker(QThread):
    """Spawns a training subprocess and streams its stdout line by line."""
    log_line = pyqtSignal(str)
    finished = pyqtSignal()
    error    = pyqtSignal(str)

    _CMD_BASE = {
        'classifier': ['-m', 'modules.cnn.train', '--model', 'classifier'],
        'keypoint':   ['-m', 'modules.cnn.train', '--model', 'keypoint'],
        'lstm':       ['-m', 'modules.lstm.train'],
        'detector':   ['-m', 'modules.detector.train'],
    }

    def __init__(self, module: str, epochs: int, lr: float, batch_size: int):
        super().__init__()
        self._module     = module
        self._epochs     = epochs
        self._lr         = lr
        self._batch_size = batch_size
        self._proc       = None

    def run(self):
        base = self._CMD_BASE.get(self._module)
        if base is None:
            self.error.emit(f"Unknown module: {self._module}")
            return

        cmd = [sys.executable] + base + [
            '--epochs',     str(self._epochs),
            '--lr',         str(self._lr),
            '--batch_size', str(self._batch_size),
        ]
        root = str(Path(__file__).parent.parent)

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=root,
            )
            for line in self._proc.stdout:
                self.log_line.emit(line.rstrip())
            self._proc.wait()
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        if self._proc:
            self._proc.terminate()
            self._proc = None
