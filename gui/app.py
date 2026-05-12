import html
import sys

import cv2
import numpy as np
from PIL import Image

import torch

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QLineEdit, QTabWidget,
    QSpinBox, QDoubleSpinBox, QComboBox,
    QFileDialog, QSizePolicy, QCheckBox,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QBrush, QPalette

DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available()  else
    "cuda" if torch.cuda.is_available()            else
    "cpu"
)

# ── Court geometry constants (all normalised 0–1) ──────────────────────────
_sw = 8.23 / 10.97          # singles/doubles width ratio
_lx = (1 - _sw) / 2         # left singles line   ≈ 0.125
_rx = 1 - _lx               # right singles line  ≈ 0.875
_ny = 0.5                    # net
_sl = 6.40 / 23.77           # service-line offset from net ≈ 0.269
_ty = _ny - _sl              # top service line    ≈ 0.231
_by = _ny + _sl              # bottom service line ≈ 0.769

# Out-of-bounds padding shown around the court diagram
# 15 ft beyond each baseline (78 ft court) and 10 ft beyond each sideline (36 ft)
_PAD_X = 10.0 / 36.0        # ≈ 0.278 — side clearance in court-width units
_PAD_Y = 15.0 / 78.0        # ≈ 0.192 — end clearance in court-length units
_TW    = 1.0 + 2 * _PAD_X   # total normalised width  ≈ 1.556
_TH    = 1.0 + 2 * _PAD_Y   # total normalised height ≈ 1.385

# Player overlay colours: BGR for OpenCV, QColor for QPainter
_CV_COLORS = [(255, 80, 80), (80, 80, 255)]
_QT_COLORS = [QColor(255, 80, 80), QColor(80, 80, 255)]


# ── helpers ────────────────────────────────────────────────────────────────

def _frame_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def _bgr_to_pixmap(frame_bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qi = QImage(rgb.tobytes(), w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qi)


# ── Court map widget ────────────────────────────────────────────────────────

class CourtMapWidget(QWidget):
    """Paints a top-down 2-D court diagram with projected player positions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 320)
        self._positions: list[tuple[float, float]] = []

    def set_players(self, positions: list[tuple[float, float]]):
        self._positions = positions
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        W, H = self.width(), self.height()
        mx, my = 12, 12
        cw, ch = W - 2 * mx, H - 2 * my

        # Map court-relative coords (padded range) to pixel coords
        def tx(x): return int(mx + (x + _PAD_X) / _TW * cw)
        def ty(y): return int(my + (y + _PAD_Y) / _TH * ch)

        # Dark background
        p.fillRect(0, 0, W, H, QColor(28, 28, 40))

        # Out-of-bounds surround (full padded area)
        p.fillRect(mx, my, cw, ch, QColor(36, 68, 36))

        # Court surface (inner court only)
        cx0, cy0 = tx(0), ty(0)
        cx1, cy1 = tx(1), ty(1)
        p.fillRect(cx0, cy0, cx1 - cx0, cy1 - cy0, QColor(44, 110, 44))

        white = QPen(QColor(255, 255, 255), 2)
        p.setPen(white)

        # Court boundary
        p.drawRect(cx0, cy0, cx1 - cx0, cy1 - cy0)

        # Singles sidelines
        p.drawLine(tx(_lx), ty(0), tx(_lx), ty(1))
        p.drawLine(tx(_rx), ty(0), tx(_rx), ty(1))

        # Service lines + centre service line
        p.drawLine(tx(_lx), ty(_ty), tx(_rx), ty(_ty))
        p.drawLine(tx(_lx), ty(_by), tx(_rx), ty(_by))
        p.drawLine(tx(0.5), ty(_ty), tx(0.5), ty(_by))

        # Net (thicker)
        p.setPen(QPen(QColor(220, 220, 220), 3))
        p.drawLine(tx(0), ty(_ny), tx(1), ty(_ny))

        # Player dots — positions may be outside [0,1] (behind baseline etc.)
        for i, (px, py) in enumerate(self._positions):
            # Only draw if within the padded canvas
            if -_PAD_X <= px <= 1 + _PAD_X and -_PAD_Y <= py <= 1 + _PAD_Y:
                c = _QT_COLORS[i % len(_QT_COLORS)]
                p.setBrush(QBrush(c))
                p.setPen(QPen(QColor(255, 255, 255), 1))
                r = 7
                p.drawEllipse(tx(px) - r, ty(py) - r, 2 * r, 2 * r)

        p.end()


# ── Main window ─────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CourtSense")
        self.resize(1280, 820)

        self._models: dict = {}

        # Video state
        self._cap           = None
        self._timer         = QTimer(self)
        self._timer.timeout.connect(self._read_frame)
        self._current_frame: np.ndarray | None = None
        self._current_pil:   Image.Image | None = None

        # Latest inference results
        self._last_boxes     = None
        self._last_scores    = None
        self._last_keypoints = None

        # Worker handles (kept to avoid GC)
        self._inf_worker      = None
        self._vlm_worker      = None
        self._train_worker    = None
        self._loader          = None

        self._setup_ui()
        self._load_models()

    # ── UI construction ────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        rv = QVBoxLayout(root)
        rv.setContentsMargins(4, 4, 4, 4)
        rv.setSpacing(4)

        # ─ top: video label + court map side-by-side ─
        top = QWidget()
        th = QHBoxLayout(top)
        th.setContentsMargins(0, 0, 0, 0)
        th.setSpacing(4)

        self._video_lbl = QLabel("Open an image or video to begin")
        self._video_lbl.setMinimumSize(640, 400)
        self._video_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._video_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_lbl.setStyleSheet("background:#111120; color:#555; font-size:13px;")

        self._court_map = CourtMapWidget()
        self._court_map.setFixedWidth(240)

        th.addWidget(self._video_lbl, stretch=1)
        th.addWidget(self._court_map)

        # ─ controls row ─
        ctrl = QHBoxLayout()
        ctrl.setSpacing(6)

        self._btn_image  = QPushButton("Open Image")
        self._btn_video  = QPushButton("Open Video")
        self._btn_camera = QPushButton("Camera")
        self._btn_play   = QPushButton("▶  Play")
        self._btn_stop   = QPushButton("■  Stop")
        self._btn_play.setCheckable(True)

        for w in (self._btn_image, self._btn_video, self._btn_camera,
                  self._btn_play, self._btn_stop):
            ctrl.addWidget(w)

        ctrl.addSpacing(12)

        self._chk_boxes = QCheckBox("Bounding boxes")
        self._chk_kpts  = QCheckBox("Court keypoints")
        self._chk_boxes.setChecked(True)
        self._chk_kpts.setChecked(True)
        ctrl.addWidget(self._chk_boxes)
        ctrl.addWidget(self._chk_kpts)
        ctrl.addStretch()

        self._btn_image.clicked.connect(self._open_image)
        self._btn_video.clicked.connect(self._open_video)
        self._btn_camera.clicked.connect(self._open_camera)
        self._btn_play.toggled.connect(self._toggle_play)
        self._btn_stop.clicked.connect(self._stop_video)
        self._chk_boxes.toggled.connect(self._render)
        self._chk_kpts.toggled.connect(self._render)

        # ─ bottom tabs ─
        tabs = QTabWidget()

        # Chat tab
        chat_w = QWidget()
        cv = QVBoxLayout(chat_w)
        cv.setContentsMargins(4, 4, 4, 4)
        cv.setSpacing(4)

        self._chat_display = QTextEdit()
        self._chat_display.setReadOnly(True)
        self._chat_display.setFont(QFont("Menlo", 11))
        self._chat_display.setStyleSheet(
            "background:#0d1117; color:#e0e0e0; font-size:12px;")
        cv.addWidget(self._chat_display, stretch=1)

        chat_row = QHBoxLayout()
        self._chat_input = QLineEdit()
        self._chat_input.setPlaceholderText("Ask something about the scene…")
        self._chat_input.returnPressed.connect(self._send_chat)
        self._btn_send     = QPushButton("Send")
        self._btn_describe = QPushButton("Describe Scene")
        self._btn_send.clicked.connect(self._send_chat)
        self._btn_describe.clicked.connect(self._describe_scene)
        chat_row.addWidget(self._chat_input, stretch=1)
        chat_row.addWidget(self._btn_send)
        chat_row.addWidget(self._btn_describe)
        cv.addLayout(chat_row)

        # Training tab
        train_w = QWidget()
        tv = QVBoxLayout(train_w)
        tv.setContentsMargins(4, 4, 4, 4)
        tv.setSpacing(4)

        train_cfg = QHBoxLayout()
        train_cfg.addWidget(QLabel("Model:"))
        self._train_model = QComboBox()
        self._train_model.addItems(["classifier", "keypoint", "lstm", "detector"])
        train_cfg.addWidget(self._train_model)

        train_cfg.addWidget(QLabel("Epochs:"))
        self._train_epochs = QSpinBox()
        self._train_epochs.setRange(1, 500)
        self._train_epochs.setValue(30)
        train_cfg.addWidget(self._train_epochs)

        train_cfg.addWidget(QLabel("LR:"))
        self._train_lr = QDoubleSpinBox()
        self._train_lr.setDecimals(6)
        self._train_lr.setRange(1e-6, 1.0)
        self._train_lr.setValue(1e-3)
        self._train_lr.setSingleStep(1e-4)
        train_cfg.addWidget(self._train_lr)

        train_cfg.addWidget(QLabel("Batch:"))
        self._train_batch = QSpinBox()
        self._train_batch.setRange(1, 256)
        self._train_batch.setValue(32)
        train_cfg.addWidget(self._train_batch)

        train_cfg.addStretch()
        self._btn_train_start = QPushButton("Start")
        self._btn_train_stop  = QPushButton("Stop")
        self._btn_train_stop.setEnabled(False)
        self._btn_train_start.clicked.connect(self._start_training)
        self._btn_train_stop.clicked.connect(self._stop_training)
        train_cfg.addWidget(self._btn_train_start)
        train_cfg.addWidget(self._btn_train_stop)
        tv.addLayout(train_cfg)

        self._train_log = QTextEdit()
        self._train_log.setReadOnly(True)
        self._train_log.setStyleSheet(
            "background:#0d1117; color:#b0b0b0; font-family:monospace; font-size:11px;")
        tv.addWidget(self._train_log, stretch=1)

        tabs.addTab(chat_w,  "VLM Chat")
        tabs.addTab(train_w, "Training")
        tabs.setMaximumHeight(260)

        # ─ assemble ─
        rv.addWidget(top, stretch=1)
        rv.addLayout(ctrl)
        rv.addWidget(tabs)

        self.statusBar().showMessage("Loading models…")

    # ── Model loading ──────────────────────────────────────────────────────

    def _load_models(self):
        from gui.workers import ModelLoader
        self._loader = ModelLoader()
        self._loader.progress.connect(self.statusBar().showMessage)
        self._loader.ready.connect(self._on_models_ready)
        self._loader.start()

    def _on_models_ready(self, models: dict):
        self._models = models
        loaded = [k for k in
                  ('classifier', 'keypoint', 'lstm', 'detector', 'vlm_model')
                  if k in models]
        self.statusBar().showMessage(
            f"Ready  ·  device: {DEVICE}  ·  loaded: {', '.join(loaded)}"
        )

    # ── Video controls ─────────────────────────────────────────────────────

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)")
        if not path:
            return
        self._stop_video()
        frame = cv2.imread(path)
        if frame is not None:
            self._process_frame(frame)

    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Videos (*.mp4 *.avi *.mov *.mkv *.m4v)")
        if not path:
            return
        self._stop_video()
        self._cap = cv2.VideoCapture(path)
        self._btn_play.setChecked(True)

    def _open_camera(self):
        self._stop_video()
        self._cap = cv2.VideoCapture(0)
        self._btn_play.setChecked(True)

    def _toggle_play(self, playing: bool):
        if playing and self._cap and self._cap.isOpened():
            self._timer.start(33)   # ~30 fps
        else:
            self._timer.stop()

    def _stop_video(self):
        self._timer.stop()
        self._btn_play.setChecked(False)
        if self._cap:
            self._cap.release()
            self._cap = None

    def _read_frame(self):
        if self._cap and self._cap.isOpened():
            ok, frame = self._cap.read()
            if ok:
                self._process_frame(frame)
            else:
                self._stop_video()

    # ── Inference pipeline ─────────────────────────────────────────────────

    def _process_frame(self, frame_bgr: np.ndarray):
        self._current_frame = frame_bgr
        self._current_pil   = _frame_to_pil(frame_bgr)
        self._render()

        # Only start a new inference pass when the previous one has finished
        if self._inf_worker is None or not self._inf_worker.isRunning():
            self._run_inference()

    def _run_inference(self):
        if self._current_pil is None:
            return
        if not self._models.get('detector') and not self._models.get('keypoint'):
            return

        from gui.workers import InferenceWorker
        w = InferenceWorker(self._models, self._current_pil)
        w.result.connect(self._on_inference)
        self._inf_worker = w
        w.start()

    def _on_inference(self, result: dict):
        if 'error' in result:
            self.statusBar().showMessage(f"Inference error: {result['error']}")
            return

        self._last_boxes     = result.get('boxes')
        self._last_scores    = result.get('scores')
        self._last_keypoints = result.get('keypoints')
        self._render()
        self._update_court_map()

    def _update_court_map(self):
        boxes_np = (self._last_boxes.numpy()
                    if self._last_boxes is not None and len(self._last_boxes) > 0
                    else None)
        kp_np = (self._last_keypoints.numpy()
                 if self._last_keypoints is not None
                 else None)

        positions: list[tuple[float, float]] = []

        if boxes_np is not None:
            if kp_np is not None:
                # Use the detected keypoint bounding box as the court region
                kp_min = kp_np.min(axis=0)
                kp_rng = (kp_np.max(axis=0) - kp_min).clip(min=1e-6)
                for box in boxes_np:
                    bx, by, bw, bh = box
                    cx = float(bx + bw / 2)
                    cy = float(by + bh)             # foot point
                    positions.append((
                        max(-_PAD_X, min(1 + _PAD_X, float((cx - kp_min[0]) / kp_rng[0]))),
                        max(-_PAD_Y, min(1 + _PAD_Y, float((cy - kp_min[1]) / kp_rng[1]))),
                    ))
            else:
                # Fallback: raw normalised frame coordinates
                for box in boxes_np:
                    bx, by, bw, bh = box
                    positions.append((float(bx + bw / 2), float(by + bh)))

        self._court_map.set_players(positions)

    # ── Rendering ──────────────────────────────────────────────────────────

    def _render(self):
        if self._current_frame is None:
            return

        frame = self._current_frame.copy()
        h, w  = frame.shape[:2]

        # Bounding boxes
        if self._chk_boxes.isChecked() and self._last_boxes is not None:
            for i, (box, sc) in enumerate(
                    zip(self._last_boxes, self._last_scores)):
                bx, by, bw, bh = box.tolist()
                x1, y1 = int(bx * w),          int(by * h)
                x2, y2 = int((bx + bw) * w),   int((by + bh) * h)
                c = _CV_COLORS[i % len(_CV_COLORS)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
                cv2.putText(frame, f"P{i + 1}  {sc:.2f}",
                            (x1, max(y1 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)

        # Court keypoints
        if self._chk_kpts.isChecked() and self._last_keypoints is not None:
            for kp in self._last_keypoints:
                kx = int(kp[0].item() * w)
                ky = int(kp[1].item() * h)
                cv2.circle(frame, (kx, ky), 4, (50, 255, 50), -1)

        pixmap = _bgr_to_pixmap(frame)
        lw, lh = self._video_lbl.width(), self._video_lbl.height()
        self._video_lbl.setPixmap(
            pixmap.scaled(lw, lh,
                          Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()

    # ── VLM chat ───────────────────────────────────────────────────────────

    def _send_chat(self):
        prompt = self._chat_input.text().strip()
        if not prompt:
            return
        if not self._models.get('vlm_model'):
            self._chat_append("System", "VLM not loaded yet.")
            return
        if self._current_pil is None:
            self._chat_append("System", "No image loaded.")
            return
        self._chat_input.clear()
        self._chat_append("You", prompt)
        self._run_vlm(prompt=prompt)

    def _describe_scene(self):
        if not self._models.get('vlm_model'):
            self._chat_append("System", "VLM not loaded yet.")
            return
        if self._current_pil is None:
            self._chat_append("System", "No image loaded.")
            return
        ctx = None
        if self._last_boxes is not None and len(self._last_boxes) > 0:
            ctx = {
                'n_players': len(self._last_boxes),
                'scores':    self._last_scores.tolist(),
            }
        self._run_vlm(context=ctx)

    def _run_vlm(self, prompt: str = "", context=None):
        if self._vlm_worker and self._vlm_worker.isRunning():
            self._chat_append("System", "VLM is already running — please wait.")
            return
        self.statusBar().showMessage("VLM thinking…")
        self._set_chat_enabled(False)

        from gui.workers import VLMWorker
        self._vlm_worker = VLMWorker(
            self._models['vlm_model'],
            self._models['vlm_processor'],
            self._current_pil,
            prompt,
            context,
        )
        self._vlm_worker.response.connect(self._on_vlm_response)
        self._vlm_worker.error.connect(
            lambda e: self._on_vlm_response(f"[Error: {e}]"))
        self._vlm_worker.start()

    def _on_vlm_response(self, text: str):
        self._chat_append("VLM", text)
        self._set_chat_enabled(True)
        loaded = [k for k in
                  ('classifier', 'keypoint', 'lstm', 'detector', 'vlm_model')
                  if k in self._models]
        self.statusBar().showMessage(
            f"Ready  ·  device: {DEVICE}  ·  loaded: {', '.join(loaded)}"
        )

    def _chat_append(self, speaker: str, text: str):
        safe = html.escape(text).replace("\n", "<br>")
        self._chat_display.append(f"<b>{speaker}:</b> {safe}")
        sb = self._chat_display.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _set_chat_enabled(self, enabled: bool):
        self._chat_input.setEnabled(enabled)
        self._btn_send.setEnabled(enabled)
        self._btn_describe.setEnabled(enabled)

    # ── Training ───────────────────────────────────────────────────────────

    def _start_training(self):
        if self._train_worker and self._train_worker.isRunning():
            return
        self._train_log.clear()
        self._btn_train_start.setEnabled(False)
        self._btn_train_stop.setEnabled(True)

        from gui.workers import TrainingWorker
        self._train_worker = TrainingWorker(
            module     = self._train_model.currentText(),
            epochs     = self._train_epochs.value(),
            lr         = self._train_lr.value(),
            batch_size = self._train_batch.value(),
        )
        self._train_worker.log_line.connect(self._train_log.append)
        self._train_worker.finished.connect(self._on_training_done)
        self._train_worker.error.connect(
            lambda e: self._train_log.append(f"ERROR: {e}"))
        self._train_worker.start()

    def _stop_training(self):
        if self._train_worker:
            self._train_worker.stop()
        self._on_training_done()

    def _on_training_done(self):
        self._btn_train_start.setEnabled(True)
        self._btn_train_stop.setEnabled(False)

    # ── Cleanup ────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._stop_video()
        for w in (self._inf_worker, self._vlm_worker, self._train_worker):
            if w and w.isRunning():
                w.quit()
                w.wait(2000)
        event.accept()


# ── Dark palette ────────────────────────────────────────────────────────────

def _dark_palette() -> QPalette:
    p = QPalette()
    bg  = QColor(28, 28, 40)
    txt = QColor(215, 215, 215)
    sel = QColor(65, 120, 210)
    btn = QColor(48, 48, 68)

    p.setColor(QPalette.ColorRole.Window,          bg)
    p.setColor(QPalette.ColorRole.WindowText,      txt)
    p.setColor(QPalette.ColorRole.Base,            QColor(18, 18, 28))
    p.setColor(QPalette.ColorRole.AlternateBase,   QColor(38, 38, 55))
    p.setColor(QPalette.ColorRole.ToolTipBase,     txt)
    p.setColor(QPalette.ColorRole.ToolTipText,     bg)
    p.setColor(QPalette.ColorRole.Text,            txt)
    p.setColor(QPalette.ColorRole.Button,          btn)
    p.setColor(QPalette.ColorRole.ButtonText,      txt)
    p.setColor(QPalette.ColorRole.BrightText,      QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.Link,            sel)
    p.setColor(QPalette.ColorRole.Highlight,       sel)
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    return p


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
