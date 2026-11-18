"""
Tennis Tracker — Streamlit GUI
Launch: streamlit run gui/gui.py
"""

import io, os, sys, time, tempfile, contextlib
from collections import deque
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as T

from modules.vlm.vlm_service import ask_vlm
from modules.detector.train import load_detector
from modules.cnn.train import load_classifier, load_keypoint_detector

# ── Constants ──────────────────────────────────────────────────────────────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
WEIGHTS_DIR_LSTM = Path(__file__).parent.parent / "modules" / "lstm" / "weights"
WEIGHTS_DIR_MLP  = Path(__file__).parent.parent / "modules" / "mlp"  / "weights"
WEIGHTS_DIR_CNN  = Path(__file__).parent.parent / "modules" / "cnn"  / "weights"

# Per-player colors (PIL RGB)
_PLAYER_COLORS = ["#00FF87", "#FF6B6B", "#FFE66D", "#A78BFA"]
_BALL_COLOR    = "#00D4FF"
_PRED_COLOR    = "#FF8C00"

st.set_page_config(page_title="TennisTracker", page_icon=None, layout="wide")

st.markdown("""
<style>
/* ── Reset ───────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stApp"] { background: #ffffff; }

.main .block-container {
    padding-top: 1rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px;
}

/* ── Sidebar ─────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #f5f5f5 !important;
    border-right: 1px solid #ddd !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 22px 18px 18px !important;
}

/* ── Navigation tabs ─────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 0 !important;
    border-bottom: 2px solid #e0e0e0;
    margin-bottom: 1.4rem;
    padding: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    color: #999 !important;
    padding: 10px 20px 9px !important;
    transition: color 0.15s !important;
}
.stTabs [aria-selected="true"] {
    border-bottom: 2px solid #1d4ed8 !important;
    color: #111 !important;
}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    color: #444 !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

/* ── Buttons ─────────────────────────────────────────── */
.stButton > button {
    font-size: 13px;
    font-weight: 600;
    border-radius: 4px;
    border: 1px solid #ccc;
    background: #fff;
    color: #333;
    padding: 7px 0;
    transition: border-color 0.12s, background 0.12s;
}
.stButton > button:hover {
    border-color: #888;
    background: #f5f5f5;
    color: #111;
}
.stButton > button[kind="primary"] {
    background: #1d4ed8;
    border-color: #1d4ed8;
    color: #fff;
}
.stButton > button[kind="primary"]:hover {
    background: #1e40af;
    border-color: #1e40af;
}

/* ── Metrics ─────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #f9f9f9;
    border: 1px solid #e0e0e0;
    border-radius: 5px;
    padding: 10px 12px !important;
}
[data-testid="metric-container"] label {
    font-size: 10px !important;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #888 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #111 !important;
}

/* ── HR ──────────────────────────────────────────────── */
hr { border-color: #e0e0e0 !important; margin: 0.8rem 0 !important; }

/* ── Captions ────────────────────────────────────────── */
.stCaption, [data-testid="stCaptionContainer"] {
    color: #888 !important;
    font-size: 11px !important;
}

/* ── Code / logs ─────────────────────────────────────── */
[data-testid="stCode"] { font-size: 11px !important; }

/* ── Custom classes ──────────────────────────────────── */
.tt-brand {
    padding-bottom: 14px;
    margin-bottom: 14px;
    border-bottom: 1px solid #ddd;
}
.tt-brand-name { font-size: 14px; font-weight: 700; color: #111; }
.tt-brand-sub  { font-size: 11px; color: #999; margin-top: 2px; }

.tt-section {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #888;
    padding: 4px 0;
    border-bottom: 1px solid #e0e0e0;
    margin-bottom: 6px;
    margin-top: 14px;
}

.tt-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 5px 0;
    border-bottom: 1px solid #f0f0f0;
}
.tt-row-label { font-size: 12px; color: #666; }
.tt-row-value { font-size: 13px; color: #111; font-weight: 600; }
.tt-row-sub   { font-size: 11px; color: #999; margin-left: 5px; font-weight: 400; }

.tt-model-status { display: flex; align-items: center; gap: 7px; padding: 3px 0; }
.tt-model-dot    { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.tt-model-label  { font-size: 12px; color: #333; }

.tt-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 380px;
    border: 1px dashed #ccc;
    border-radius: 8px;
    gap: 8px;
    background: #fafafa;
}
.tt-empty-icon  { margin-bottom: 6px; opacity: 0.35; }
.tt-empty-title { font-size: 14px; font-weight: 600; color: #555; }
.tt-empty-sub   { font-size: 12px; color: #999; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
_DEFAULTS = {
    "logs": [],
    "base_image": None,
    "detector_boxes": [], "detector_scores": [], "detector_labels": [],
    "keypoints": [],
    "player_count": 0,
    "classifier_label": "Not run", "classifier_confidence": 0.0,
    "mlp_label": "Not run",        "mlp_confidence": 0.0,
    "messages": [],
    "trajectory_history": [],
    "trajectory_prediction": None,
    # video
    "video_output_path": None,   # path to rendered output MP4
    "video_stats":       [],     # list[dict] per-frame stats
    "video_processed":   False,
    "video_frame":       None,   # BytesIO JPEG of a representative frame for VLM
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def add_log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{ts}] [{level}] {msg}")


# ── Model cache ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_detector():
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    return load_detector(checkpoint="best", device=device), device

@st.cache_resource
def _check_vlm():
    """Test VLM once per server session — not on every rerun."""
    try:
        import ollama
        from modules.vlm.vlm_service import VLM_MODEL
        ollama.chat(model=VLM_MODEL, messages=[{"role":"user","content":"ping"}])
        return True
    except Exception:
        return False

@st.cache_resource
def get_classifier():
    return load_classifier(weights="best").eval()

@st.cache_resource
def get_keypoint_detector():
    return load_keypoint_detector(weights="best").eval()

@st.cache_resource
def get_mlp():
    path = WEIGHTS_DIR_MLP / "mlp_best.pt"
    if not path.exists():
        return None
    from modules.mlp.train import load_mlp
    return load_mlp(weights="best").eval()

@st.cache_resource
def get_lstm():
    path = WEIGHTS_DIR_LSTM / "lstm_best.pt"
    if not path.exists():
        return None
    from modules.lstm.train import load_lstm
    return load_lstm(checkpoint="best").eval()


# ── Inference ──────────────────────────────────────────────────────────────────
_BALL_SCORE_THRESH = 0.10   # balls are small — run at a low threshold

# Court calibration — fraction of frame height where the far/near baselines sit.
# The stands fill the top and bottom of the frame, so raw image_y/h never reaches
# 0 or 1 at the baselines.  Tune these two numbers if player depth looks off.
_COURT_IMG_Y_FAR  = 0.22   # far baseline  (top end of court in the broadcast frame)
_COURT_IMG_Y_NEAR = 0.86   # near baseline (bottom end of court in the broadcast frame)

def _img_to_court(cx_norm, cy_norm):
    """Remap normalised image coords → court [0,1] coords, clipped to court bounds."""
    court_x = cx_norm   # lateral: image_x already spans the court width reasonably
    span    = _COURT_IMG_Y_NEAR - _COURT_IMG_Y_FAR
    court_y = (cy_norm - _COURT_IMG_Y_FAR) / span
    return court_x, max(0.0, min(1.0, court_y))

def run_detector_on_image(pil_image, confidence_threshold):
    detector, device = get_detector()
    w, h = pil_image.size
    tf = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    tensor = tf(pil_image).unsqueeze(0).to(device)
    # Run with the lower of player/ball thresholds so we catch balls
    raw_thresh = min(confidence_threshold / 100, _BALL_SCORE_THRESH)
    dets   = detector.detect(tensor, score_threshold=raw_thresh)
    boxes  = dets[0]["boxes"].cpu().tolist()
    scores = dets[0]["scores"].cpu().tolist()
    labels = dets[0]["labels"].cpu().tolist()
    sx, sy = w / 640, h / 640

    # Minimum player box height: 8% of 640-px input (filters ball kids / crowd)
    min_box_h = 640 * 0.08

    players, balls = [], []
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        if label == 1 and score >= confidence_threshold / 100:
            if (y2 - y1) >= min_box_h:   # size gate
                players.append((box, score))
        elif label == 2 and score >= _BALL_SCORE_THRESH:
            balls.append((box, score))

    # Keep top 2 players (highest confidence), top 2 ball candidates
    players = sorted(players, key=lambda t: t[1], reverse=True)[:2]
    balls   = sorted(balls,   key=lambda t: t[1], reverse=True)[:2]

    fb, fs, fl = [], [], []
    for box, score in players:
        x1, y1, x2, y2 = box
        fb.append([x1*sx, y1*sy, x2*sx, y2*sy]); fs.append(score); fl.append(1)
    for box, score in balls:
        x1, y1, x2, y2 = box
        fb.append([x1*sx, y1*sy, x2*sx, y2*sy]); fs.append(score); fl.append(2)
    return fb, fs, fl

def run_classifier_on_image(pil_image):
    model = get_classifier()
    tf = T.Compose([T.Resize((128,128)), T.ToTensor(),
                    T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)])
    with torch.no_grad():
        logits = model(tf(pil_image).unsqueeze(0))
        probs  = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
    return model.CLASSES[pred.item()], conf.item()

def run_mlp_on_image(pil_image):
    model = get_mlp()
    if model is None:
        return "Not trained", 0.0
    tf = T.Compose([T.Resize((32,32)), T.Grayscale(), T.ToTensor(),
                    T.Normalize([0.5],[0.5]), T.Lambda(lambda x: x.flatten())])
    with torch.no_grad():
        logits = model(tf(pil_image).unsqueeze(0))
        probs  = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
    return model.CLASSES[pred.item()], conf.item()

def run_keypoint_on_image(pil_image):
    model = get_keypoint_detector()
    w, h = pil_image.size
    tf = T.Compose([T.Resize((320,320)), T.ToTensor(),
                    T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)])
    with torch.no_grad():
        out = model(tf(pil_image).unsqueeze(0))
    points = out.squeeze().cpu().tolist()
    return [[x*w, y*h] for x, y in points]

def run_lstm_prediction(trajectory):
    model = get_lstm()
    if model is None or len(trajectory) < 8:
        return None
    seed = torch.tensor(trajectory[-8:], dtype=torch.float32)
    pred = model.predict_sequence(seed, steps=1)
    return pred[0].tolist()


# ── SimpleTracker ──────────────────────────────────────────────────────────────
class SimpleTracker:
    """IoU-based tracker with consistent integer IDs across frames."""

    def __init__(self, iou_thresh=0.25, max_lost=15, max_tracks=None):
        self.tracks     = {}
        self.next_id    = 0
        self.thresh     = iou_thresh
        self.max_lost   = max_lost
        self.max_tracks = max_tracks

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0],b[0]), max(a[1],b[1])
        ix2, iy2 = min(a[2],b[2]), min(a[3],b[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        if inter == 0:
            return 0.0
        return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

    def update(self, dets: list) -> dict:
        """dets: list[[x1,y1,x2,y2]]. Returns {id: box}."""
        if not dets:
            for tid in list(self.tracks):
                self.tracks[tid]['lost'] += 1
                if self.tracks[tid]['lost'] > self.max_lost:
                    del self.tracks[tid]
            return {tid: t['box'] for tid, t in self.tracks.items()}

        if not self.tracks:
            for d in dets:
                if self.max_tracks and len(self.tracks) >= self.max_tracks:
                    break
                self.tracks[self.next_id] = {'box': d, 'lost': 0}
                self.next_id += 1
            return {tid: t['box'] for tid, t in self.tracks.items()}

        matched_d = set()
        for tid in list(self.tracks):
            best_iou, best_i = self.thresh, -1
            for i, d in enumerate(dets):
                if i in matched_d: continue
                iou = self._iou(self.tracks[tid]['box'], d)
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_i >= 0:
                self.tracks[tid] = {'box': dets[best_i], 'lost': 0}
                matched_d.add(best_i)
            else:
                self.tracks[tid]['lost'] += 1
                if self.tracks[tid]['lost'] > self.max_lost:
                    del self.tracks[tid]

        for i, d in enumerate(dets):
            if i not in matched_d:
                if self.max_tracks and len(self.tracks) >= self.max_tracks:
                    continue
                self.tracks[self.next_id] = {'box': d, 'lost': 0}
                self.next_id += 1

        return {tid: t['box'] for tid, t in self.tracks.items()}


# ── PIL annotation (fast, no transparency math) ────────────────────────────────
def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def annotate_frame(
    pil_image:     Image.Image,
    player_tracks: dict,          # {rank: box}  rank = display order
    ball_tracks:   dict,          # {id: box}
    player_trails: dict,          # {rank: list[(x_n, y_n)]}
    ball_trail:    list,          # list[(x_n, y_n)]
    lstm_pred:     list | None,
    frame_idx:     int,
    total_frames:  int,
) -> Image.Image:
    """Draw detection overlays using PIL — fast and dependency-light."""
    out  = pil_image.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    iw, ih = out.size

    def n2px(x, y):
        return (int(x * iw), int(y * ih))

    # ── Trails ──────────────────────────────────────────────────────────
    for rank, trail in player_trails.items():
        if len(trail) < 2:
            continue
        hex_c  = _PLAYER_COLORS[rank % len(_PLAYER_COLORS)]
        rgb    = _hex_to_rgb(hex_c)
        pts_px = [n2px(*p) for p in trail]
        for j in range(1, len(pts_px)):
            alpha = int(60 + 180 * (j / len(pts_px)))
            draw.line([pts_px[j-1], pts_px[j]], fill=rgb + (alpha,), width=3)

    if len(ball_trail) >= 2:
        ball_rgb = _hex_to_rgb(_BALL_COLOR)
        bpts = [n2px(*p) for p in ball_trail]
        for j in range(1, len(bpts)):
            alpha = int(60 + 180 * (j / len(bpts)))
            draw.line([bpts[j-1], bpts[j]], fill=ball_rgb + (alpha,), width=2)

    # ── Player boxes ────────────────────────────────────────────────────
    for rank, box in player_tracks.items():
        x1, y1, x2, y2 = [int(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        label = f" P{rank+1} "
        bbox  = draw.textbbox((x1, y1), label)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        draw.rectangle([x1, max(0, y1-th-6), x1+tw+4, y1], fill=(255,0,0,220))
        draw.text((x1+2, max(0, y1-th-3)), label, fill=(255,255,255,255))


    # ── Ball ────────────────────────────────────────────────────────────
    for _, bbox in ball_tracks.items():
        bx = int((bbox[0]+bbox[2])/2); by = int((bbox[1]+bbox[3])/2)
        ball_rgb = _hex_to_rgb(_BALL_COLOR)
        r = 10
        draw.ellipse([bx-r, by-r, bx+r, by+r], fill=ball_rgb+(220,), outline=(0,0,0,200), width=1)
        draw.line([(bx-r-4,by),(bx+r+4,by)], fill=ball_rgb+(180,), width=1)
        draw.line([(bx,by-r-4),(bx,by+r+4)], fill=ball_rgb+(180,), width=1)
        draw.text((bx+r+4, by-8), "BALL", fill=ball_rgb+(220,))

    # ── LSTM prediction ─────────────────────────────────────────────────
    if lstm_pred:
        pred_rgb = _hex_to_rgb(_PRED_COLOR)
        px_c, py_c = n2px(*lstm_pred)
        # Diamond
        pts = [(px_c, py_c-10), (px_c+7, py_c), (px_c, py_c+10), (px_c-7, py_c)]
        draw.polygon(pts, fill=pred_rgb+(220,), outline=(255,255,255,200))
        draw.text((px_c+10, py_c-6), "PRED", fill=pred_rgb+(220,))
        # Dashed line to nearest player trail end
        if player_trails:
            best_rank = next(iter(player_trails))
            t = player_trails[best_rank]
            if t:
                lx, ly = n2px(*t[-1])
                # Draw 6 dash segments
                for k in range(6):
                    t0 = k / 6; t1 = (k+0.45) / 6
                    a = (int(lx + t0*(px_c-lx)), int(ly + t0*(py_c-ly)))
                    b = (int(lx + t1*(px_c-lx)), int(ly + t1*(py_c-ly)))
                    draw.line([a, b], fill=pred_rgb+(180,), width=2)

    # ── HUD bar ─────────────────────────────────────────────────────────
    hud_h = 36
    draw.rectangle([0, 0, iw, hud_h], fill=(10,10,10,170))
    n_p = len(player_tracks); n_b = len(ball_tracks)
    hud = (f"  FRAME {frame_idx:03d}/{total_frames:03d}"
           f"  |  PLAYERS: {n_p}"
           f"  |  BALL: {'TRACKED' if n_b else '—'}"
           f"  |  AI ACTIVE")
    draw.text((8, 10), hud, fill=(220,220,220,255))
    # Live dot
    draw.ellipse([iw-24, 13, iw-12, 25], fill=(0,255,100,255))

    return out


# ── Court mini-map (OpenCV) ────────────────────────────────────────────────────
_CM_W, _CM_H = 280, 440

def _bgr(hex_rgb):
    r,g,b = _hex_to_rgb(hex_rgb)
    return (b, g, r)

def _dashed_line_cv(img, p1, p2, color, thickness=1, gap=8):
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    L = max(1, int(np.hypot(dx, dy)))
    n = max(1, L//gap)
    for k in range(n):
        if k % 2 == 0:
            t0 = k/n; t1 = min(1.0,(k+1)/n)
            a = (int(p1[0]+t0*dx), int(p1[1]+t0*dy))
            b = (int(p1[0]+t1*dx), int(p1[1]+t1*dy))
            cv2.line(img, a, b, color, thickness)


def render_court_minimap(
    player_trails:   dict,          # {rank: list[(x_n,y_n)]}
    ball_trail:      list,
    lstm_pred:       list | None,
    current_players: list,          # [(x_n,y_n), ...]
    current_ball:    tuple | None,
) -> bytes:
    """Render a compact OpenCV court. Returns JPEG bytes."""
    img = np.zeros((_CM_H, _CM_W, 3), dtype=np.uint8)
    img[:] = (168, 99, 37)   # BGR hard-court blue

    PX = int(_CM_W * 0.08); PY = int(_CM_H * 0.12)
    CW = _CM_W - 2*PX;      CH = _CM_H - 2*PY

    def n2p(x, y):
        return (int(PX + float(x)*CW), int(PY + float(y)*CH))

    W = (255,255,255)
    A  = 4.5/36          # alley width fraction
    S  = 21/78           # service-line distance from net (normalized)

    cv2.rectangle(img, n2p(0,0), n2p(1,1), W, 2)
    # Sidelines (vertical, marking singles court width)
    cv2.line(img, n2p(A,0),   n2p(A,1),   W, 1)
    cv2.line(img, n2p(1-A,0), n2p(1-A,1), W, 1)
    # Net — horizontal across middle
    cv2.line(img, n2p(0,0.5), n2p(1,0.5), W, 2)
    # Service lines (horizontal)
    cv2.line(img, n2p(A,0.5-S),   n2p(1-A,0.5-S), W, 1)
    cv2.line(img, n2p(A,0.5+S),   n2p(1-A,0.5+S), W, 1)
    # Center service line (vertical, between service lines only)
    cv2.line(img, n2p(0.5,0.5-S), n2p(0.5,0.5+S), W, 1)
    np_pos = n2p(0.5, 0.5)
    cv2.putText(img, "NET", (np_pos[0]-14, np_pos[1]-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, W, 1)

    # Player trails
    for rank, trail in player_trails.items():
        c   = _bgr(_PLAYER_COLORS[rank % len(_PLAYER_COLORS)])
        pts = [n2p(*p) for p in trail]
        for j in range(1, len(pts)):
            thick = max(1, int(3 * j / max(len(pts),1)))
            cv2.line(img, pts[j-1], pts[j], c, thick)

    # Ball trail
    bpts = [n2p(*p) for p in ball_trail]
    ball_c = _bgr(_BALL_COLOR)
    for j in range(1, len(bpts)):
        thick = max(1, int(2 * j / max(len(bpts),1)))
        cv2.line(img, bpts[j-1], bpts[j], ball_c, thick)

    # Current player dots
    for rank, pos in enumerate(current_players):
        c   = _bgr(_PLAYER_COLORS[rank % len(_PLAYER_COLORS)])
        dot = n2p(*pos)
        cv2.circle(img, dot, 8, c, -1)
        cv2.circle(img, dot, 8, W, 1)
        cv2.putText(img, f"P{rank+1}", (dot[0]+10, dot[1]+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, c, 1)

    # Ball dot
    if current_ball:
        bdot = n2p(*current_ball)
        cv2.circle(img, bdot, 5, ball_c, -1)
        cv2.circle(img, bdot, 5, W, 1)

    # LSTM prediction
    if lstm_pred:
        pp = n2p(*lstm_pred)
        pred_c = _bgr(_PRED_COLOR)
        diamond = np.array([
            (pp[0], pp[1]-8),(pp[0]+6,pp[1]),
            (pp[0], pp[1]+8),(pp[0]-6,pp[1])
        ], np.int32)
        cv2.fillPoly(img, [diamond], pred_c)
        cv2.polylines(img, [diamond], True, W, 1)
        if current_players:
            lp = n2p(*current_players[0])
            _dashed_line_cv(img, lp, pp, pred_c, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=88)
    return buf.getvalue()


# ── Video processing → output MP4 ─────────────────────────────────────────────
# Layout: [annotated video (left) | court minimap (right)] — no divider array,
# so combined width == VideoWriter width exactly.

_OUT_H   = 540
_OUT_W   = 900
_OUT_FPS  = 12.0


def _render_court_bgr(
    player_trails:   dict,
    ball_trail:      list,
    lstm_pred:       list | None,
    current_players: list,
    current_ball:    tuple | None,
    w: int = _CM_W, h: int = _CM_H,
) -> np.ndarray:
    """Same as render_court_minimap but returns BGR numpy array (for VideoWriter)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (168, 99, 37)
    PX = int(w*0.08); PY = int(h*0.12)
    CW = w-2*PX;      CH = h-2*PY

    def n2p(x, y):
        return (int(PX+float(x)*CW), int(PY+float(y)*CH))

    W = (255,255,255); A = 4.5/36; S = 21/78
    cv2.rectangle(img, n2p(0,0), n2p(1,1), W, 2)
    cv2.line(img, n2p(A,0),   n2p(A,1),   W, 1)
    cv2.line(img, n2p(1-A,0), n2p(1-A,1), W, 1)
    cv2.line(img, n2p(0,0.5), n2p(1,0.5), W, 2)
    cv2.line(img, n2p(A,0.5-S),   n2p(1-A,0.5-S), W, 1)
    cv2.line(img, n2p(A,0.5+S),   n2p(1-A,0.5+S), W, 1)
    cv2.line(img, n2p(0.5,0.5-S), n2p(0.5,0.5+S), W, 1)
    np_pos = n2p(0.5, 0.5)
    cv2.putText(img,"NET",(np_pos[0]-14,np_pos[1]-6),cv2.FONT_HERSHEY_SIMPLEX,0.35,W,1)

    for rank, trail in player_trails.items():
        c   = _bgr(_PLAYER_COLORS[rank % len(_PLAYER_COLORS)])
        pts = [n2p(*p) for p in trail]
        for j in range(1, len(pts)):
            cv2.line(img, pts[j-1], pts[j], c, max(1, int(3*j/max(len(pts),1))))
    bc = _bgr(_BALL_COLOR)
    bpts = [n2p(*p) for p in ball_trail]
    for j in range(1, len(bpts)):
        cv2.line(img, bpts[j-1], bpts[j], bc, max(1, int(2*j/max(len(bpts),1))))
    for rank, pos in enumerate(current_players):
        c = _bgr(_PLAYER_COLORS[rank % len(_PLAYER_COLORS)])
        dot = n2p(*pos)
        cv2.circle(img, dot, 8, c, -1); cv2.circle(img, dot, 8, W, 1)
        cv2.putText(img,f"P{rank+1}",(dot[0]+10,dot[1]+4),cv2.FONT_HERSHEY_SIMPLEX,0.32,c,1)
    if current_ball:
        bdot = n2p(*current_ball)
        cv2.circle(img, bdot, 5, bc, -1); cv2.circle(img, bdot, 5, W, 1)
    if lstm_pred:
        pp   = n2p(*lstm_pred)
        pred = _bgr(_PRED_COLOR)
        dm   = np.array([(pp[0],pp[1]-8),(pp[0]+6,pp[1]),(pp[0],pp[1]+8),(pp[0]-6,pp[1])],np.int32)
        cv2.fillPoly(img,[dm],pred); cv2.polylines(img,[dm],True,W,1)
        if current_players:
            lp = n2p(*current_players[0])
            _dashed_line_cv(img, lp, pp, pred, 1)
    return img   # BGR


_FRAME_SKIP = 3    # process every Nth frame — balances quality vs processing time
_MAX_FRAMES = 500  # cap total processed frames per video


def process_video(video_file, confidence_threshold):
    """
    Run detection + tracking on every _FRAME_SKIP-th frame, draw overlays,
    write a side-by-side MP4 (annotated video | court minimap), return
    (output_path, stats_list).
    """
    video_file.seek(0)
    suffix = Path(video_file.name).suffix if hasattr(video_file, "name") else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(video_file.read()); tmp_in = f.name

    out_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = out_tmp.name; out_tmp.close()

    player_tracker = SimpleTracker(iou_thresh=0.25, max_lost=15, max_tracks=2)
    ball_tracker   = SimpleTracker(iou_thresh=0.20, max_lost=5)
    player_trails_by_id = {}
    ball_trail          = deque(maxlen=50)
    lstm_traj           = []
    stats               = []

    try:
        cap      = cv2.VideoCapture(tmp_in)
        src_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_fps  = max(6.0, min(24.0, src_fps / _FRAME_SKIP))
        n_est    = max(1, min(_MAX_FRAMES, total // _FRAME_SKIP))

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(out_path, fourcc, out_fps, (_OUT_W, _OUT_H))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, out_fps, (_OUT_W, _OUT_H))
        prog  = st.progress(0.0, text="Building annotated video…")
        f_idx = 0; proc = 0

        while cap.isOpened() and proc < _MAX_FRAMES:
            ret, bgr = cap.read()
            if not ret:
                break

            if f_idx % _FRAME_SKIP == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                iw, ih = pil.size

                # Detection
                boxes, scores, labels = run_detector_on_image(pil, confidence_threshold)
                p_boxes = [b for b,s,l in zip(boxes,scores,labels) if l==1]
                b_boxes = [b for b,s,l in zip(boxes,scores,labels) if l==2]

                p_tracks = player_tracker.update(p_boxes)
                b_tracks = ball_tracker.update(b_boxes)

                # Trails + LSTM feed — stored in image coords (cx=x/w, cy=y/h)
                for tid, box in p_tracks.items():
                    cx = (box[0]+box[2])/(2*iw); cy = (box[1]+box[3])/(2*ih)
                    if tid not in player_trails_by_id:
                        player_trails_by_id[tid] = deque(maxlen=25)
                    player_trails_by_id[tid].append((cx, cy))
                    lstm_traj.append([cx, cy])
                for _, bbox in b_tracks.items():
                    ball_trail.append(((bbox[0]+bbox[2])/(2*iw), (bbox[1]+bbox[3])/(2*ih)))

                lstm_pred = run_lstm_prediction(lstm_traj) if len(lstm_traj) >= 8 else None

                sorted_tids          = sorted(p_tracks.keys())
                rank_by_tid          = {tid: i for i, tid in enumerate(sorted_tids)}
                player_tracks_ranked = {rank_by_tid[tid]: box for tid, box in p_tracks.items()}
                player_trails_ranked = {rank_by_tid[tid]: list(player_trails_by_id[tid])
                                        for tid in p_tracks if tid in player_trails_by_id}
                cur_players = [((box[0]+box[2])/(2*iw), (box[1]+box[3])/(2*ih))
                               for box in p_tracks.values()]
                cur_ball = None
                if b_tracks:
                    b0 = next(iter(b_tracks.values()))
                    cur_ball = ((b0[0]+b0[2])/(2*iw), (b0[1]+b0[3])/(2*ih))

                # ── Annotated frame — uses image coords as-is ──────────
                ann_pil = annotate_frame(
                    pil, player_tracks_ranked, b_tracks,
                    player_trails_ranked, list(ball_trail),
                    lstm_pred, proc+1, n_est,
                )
                ann_bgr = cv2.cvtColor(np.array(ann_pil), cv2.COLOR_RGB2BGR)
                ann_bgr = cv2.resize(ann_bgr, (_OUT_W, _OUT_H))
                writer.write(ann_bgr)

                stats.append({
                    "n_players": len(p_tracks),
                    "n_balls":   len(b_tracks),
                    "lstm_pred": lstm_pred,
                })

                # Save a frame near the middle of the video for VLM context
                if proc == max(1, n_est // 2):
                    _buf = io.BytesIO()
                    ann_pil.save(_buf, format="JPEG", quality=85)
                    _buf.seek(0)
                    st.session_state.video_frame = _buf

                proc += 1

                if proc % 5 == 0 or proc == 1:
                    prog.progress(min(proc/n_est, 1.0),
                                  text=f"Rendering frame {proc}/{n_est}…")
            f_idx += 1

        cap.release()
        writer.release()
        prog.empty()

    finally:
        os.unlink(tmp_in)

    st.session_state.trajectory_history    = lstm_traj[-100:]
    st.session_state.trajectory_prediction = (run_lstm_prediction(lstm_traj)
                                               if len(lstm_traj)>=8 else None)
    add_log(f"Video rendered: {proc} frames → {out_path}")
    return out_path, stats


# ── Full-image analysis ────────────────────────────────────────────────────────
def run_full_analysis(pil_image, confidence_threshold):
    boxes, scores, labels = run_detector_on_image(pil_image, confidence_threshold)
    cnn_label, cnn_conf   = run_classifier_on_image(pil_image)
    mlp_label, mlp_conf   = run_mlp_on_image(pil_image)
    keypoints             = run_keypoint_on_image(pil_image)

    st.session_state.detector_boxes        = boxes
    st.session_state.detector_scores       = scores
    st.session_state.detector_labels       = labels
    st.session_state.keypoints             = keypoints
    st.session_state.player_count          = len(boxes)
    st.session_state.classifier_label      = cnn_label
    st.session_state.classifier_confidence = cnn_conf
    st.session_state.mlp_label             = mlp_label
    st.session_state.mlp_confidence        = mlp_conf
    iw, ih = pil_image.size
    for box in boxes:
        x1, y1, x2, y2 = box
        st.session_state.trajectory_history.append([((x1+x2)/2)/iw, ((y1+y2)/2)/ih])
    st.session_state.trajectory_prediction = run_lstm_prediction(
        st.session_state.trajectory_history
    )
    add_log(f"Analysis done — {len(boxes)} objects | "
            f"CNN={cnn_label}({cnn_conf:.2f}) MLP={mlp_label}({mlp_conf:.2f})")


# ── Matplotlib court (court tab) ───────────────────────────────────────────────
def build_court_image(trajectory_history, trajectory_prediction, current_boxes, img_w, img_h):
    fig, ax = plt.subplots(figsize=(4.5, 7.5), dpi=100)
    fig.patch.set_facecolor("#0e1a2e")
    ax.set_facecolor("#1565a8")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)   # flip so y=0 (far baseline) is at top
    ax.set_aspect("auto")
    ax.axis("off")

    # Court geometry
    A = 4.5 / 36   # alley width fraction
    S = 21 / 78    # service-line distance from net (normalized)

    lkw = dict(color="white", solid_capstyle="round", zorder=2)

    # Baselines (horizontal, y=0 far, y=1 near)
    ax.plot([0, 1], [0, 0], lw=2.5, **lkw)
    ax.plot([0, 1], [1, 1], lw=2.5, **lkw)
    # Side tramlines (vertical)
    ax.plot([0, 0], [0, 1], lw=2.5, **lkw)
    ax.plot([1, 1], [0, 1], lw=2.5, **lkw)
    # Singles sidelines (vertical)
    ax.plot([A, A],     [0, 1], lw=1.5, **lkw)
    ax.plot([1-A, 1-A], [0, 1], lw=1.5, **lkw)
    # Net (horizontal at y=0.5)
    ax.plot([0, 1], [0.5, 0.5], lw=3.5, **lkw)
    # Service lines (horizontal)
    ax.plot([A, 1-A], [0.5 - S, 0.5 - S], lw=1.5, **lkw)
    ax.plot([A, 1-A], [0.5 + S, 0.5 + S], lw=1.5, **lkw)
    # Center service line (vertical, between service lines)
    ax.plot([0.5, 0.5], [0.5 - S, 0.5 + S], lw=1.5, **lkw)

    # Labels
    ax.text(0.5, 0.5, "NET", ha="center", va="center",
            fontsize=7, color="white", fontweight="bold", alpha=0.85, zorder=3)
    ax.text(0.5, 0.01, "Far baseline", ha="center", va="top",
            fontsize=6, color="white", alpha=0.55)
    ax.text(0.5, 0.99, "Near baseline", ha="center", va="bottom",
            fontsize=6, color="white", alpha=0.55)

    # Remap raw image coords → court [0,1] using baseline calibration constants.
    court_pts = [_img_to_court(p[0], p[1]) for p in trajectory_history]
    hx = [p[0] for p in court_pts]   # court_x (horizontal, lateral)
    hy = [p[1] for p in court_pts]   # court_y (vertical, depth)
    n  = len(hx)
    # Draw as faded scatter dots (not connected lines) — trajectory_history can mix
    # multiple players/frames, so connecting points creates misleading zigzag lines.
    if n > 0:
        alphas = [0.15 + 0.75 * (i / max(n - 1, 1)) for i in range(n)]
        sizes  = [15   + 45   * (i / max(n - 1, 1)) for i in range(n)]
        for i, (x, y, a, s) in enumerate(zip(hx, hy, alphas, sizes)):
            ax.scatter(x, y, s=s, c=_BALL_COLOR, zorder=4, alpha=a,
                       edgecolors="none")
        # Bright dot for most-recent position
        ax.scatter(hx[-1], hy[-1], s=90, c=_BALL_COLOR, zorder=6,
                   edgecolors="white", lw=1.2)

    # Current player / ball dots from detector boxes
    if img_w > 0 and img_h > 0:
        player_idx = 0
        for i, (box) in enumerate(current_boxes):
            label = st.session_state.detector_labels[i]
            x1, y1, x2, y2 = box
            plot_x, plot_y = _img_to_court(((x1+x2)/2)/img_w, ((y1+y2)/2)/img_h)
            if label == 1:
                c = _PLAYER_COLORS[player_idx % len(_PLAYER_COLORS)]
                ax.scatter(plot_x, plot_y, s=120, c=c, zorder=5,
                           edgecolors="white", lw=1.5)
                ax.text(plot_x, plot_y + 0.04, f"P{player_idx+1}",
                        ha="center", fontsize=6.5, color=c, zorder=5)
                player_idx += 1
            elif label == 2:
                ax.scatter(plot_x, plot_y, s=80, c=_BALL_COLOR, marker="o",
                           zorder=5, edgecolors="white", lw=1.2)

    # LSTM prediction — remap raw image coords before plotting
    if trajectory_prediction:
        px, py = _img_to_court(trajectory_prediction[0], trajectory_prediction[1])
        ax.scatter(px, py, s=150, c=_PRED_COLOR, marker="*",
                   zorder=7, edgecolors="white", lw=1.2, label="next")
        if n > 0:
            ax.plot([hx[-1], px], [hy[-1], py], "--",
                    color=_PRED_COLOR, lw=1.5, alpha=0.85, zorder=4)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.03)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=100)
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()


# ── Training helper ────────────────────────────────────────────────────────────
def _capture_and_train(fn, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            fn(**kwargs)
        except Exception as e:
            print(f"\nERROR: {e}")
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div style="display:flex;align-items:center;gap:10px;padding:2px 0 2px;margin-bottom:6px">'
    '<span style="font-size:14px;font-weight:700;color:#111">TennisTracker</span>'
    '<div style="width:1px;height:13px;background:#ddd;margin:0 4px;flex-shrink:0"></div>'
    '<span style="font-size:11px;color:#999">Faster R-CNN · CNN · MLP · LSTM · VLM</span>'
    '</div>',
    unsafe_allow_html=True,
)

lstm_ready = (WEIGHTS_DIR_LSTM / "lstm_best.pt").exists()
mlp_ready  = (WEIGHTS_DIR_MLP  / "mlp_best.pt").exists()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
_det_ok  = (Path(__file__).parent.parent / "modules/detector/weights/detector_best.pt").exists()
_cnn_ok  = (WEIGHTS_DIR_CNN / "classifier_best.pt").exists()
_lstm_ok = lstm_ready
_mlp_ok  = mlp_ready

# Brand
st.sidebar.markdown(
    '<div class="tt-brand">'
    '<div class="tt-brand-name">TennisTracker</div>'
    '<div class="tt-brand-sub">AI Analysis System · USJ</div>'
    '</div>',
    unsafe_allow_html=True,
)

# Model status
_models = [("Detector", _det_ok), ("CNN", _cnn_ok), ("MLP", _mlp_ok), ("LSTM", _lstm_ok)]
rows = "".join(
    f'<div class="tt-model-status">'
    f'<div class="tt-model-dot" style="background:{"#16a34a" if ok else "#d1d5db"}"></div>'
    f'<span class="tt-model-label" style="color:{"#222" if ok else "#aaa"}">{name}</span>'
    f'</div>'
    for name, ok in _models
)
st.sidebar.markdown(
    f'<div style="margin-bottom:6px">{rows}</div>',
    unsafe_allow_html=True,
)

# INPUT section
st.sidebar.markdown('<div class="tt-section" style="margin-top:18px">Input Mode</div>', unsafe_allow_html=True)
input_mode = st.sidebar.radio("", ["Image", "Video", "Camera"], label_visibility="collapsed")

uploaded_image = None
uploaded_video = None

if input_mode == "Image":
    uploaded_image = st.sidebar.file_uploader("", type=["png","jpg","jpeg"],
                                              label_visibility="collapsed")
elif input_mode == "Video":
    uploaded_video = st.sidebar.file_uploader("", type=["mp4","avi","mov"],
                                              label_visibility="collapsed")

# DETECTION section
st.sidebar.markdown('<div class="tt-section">Detection</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div style="font-size:11px;color:#666;margin-bottom:4px">Confidence threshold</div>',
    unsafe_allow_html=True,
)
confidence_threshold = st.sidebar.slider("", 0, 100, 50, label_visibility="collapsed")

run_model = False
if input_mode in ("Image", "Camera"):
    st.sidebar.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
    run_model = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

st.sidebar.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<hr style="border-color:rgba(255,255,255,0.05);margin:0 0 10px">', unsafe_allow_html=True)

if st.sidebar.button("Clear Data", use_container_width=True):
    if st.session_state.video_output_path:
        try: os.unlink(st.session_state.video_output_path)
        except OSError: pass
    st.session_state.trajectory_history    = []
    st.session_state.trajectory_prediction = None
    st.session_state.video_output_path     = None
    st.session_state.video_stats           = []
    st.session_state.video_processed       = False
    st.session_state.video_frame           = None
    add_log("Cleared trajectory and video data")


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
preview_tab, court_tab, chat_tab, training_tab, logs_tab = st.tabs(
    ["Tracking", "Court View", "Chat", "Training", "Logs"]
)


# ─────────────────────────────────────────────
# TAB 1 — MEDIA & TRACKING
# ─────────────────────────────────────────────
with preview_tab:

    # ── IMAGE MODE ────────────────────────────────────────────────────────
    if input_mode == "Image":
        # Run analysis BEFORE rendering columns so session state is fresh for both panels
        if uploaded_image is not None:
            uploaded_image.seek(0)
            image = Image.open(uploaded_image).convert("RGB")
            st.session_state.base_image = image
            if run_model:
                with st.spinner("Running all five AI modules…"):
                    run_full_analysis(image, confidence_threshold)

        left_col, right_col = st.columns([3, 1], gap="large")

        with right_col:
            n_players = sum(1 for l in st.session_state.detector_labels if l == 1)
            n_balls   = sum(1 for l in st.session_state.detector_labels if l == 2)

            # DETECTIONS
            st.markdown('<div class="tt-section" style="margin-top:0">Detections</div>', unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            col_a.metric("Players", n_players)
            col_b.metric("Ball", "Yes" if n_balls else "—")

            # CLASSIFICATION
            cnn_lbl  = st.session_state.classifier_label
            cnn_conf = st.session_state.classifier_confidence
            mlp_lbl  = st.session_state.mlp_label
            mlp_conf = st.session_state.mlp_confidence
            cnn_sub  = f"{cnn_conf:.0%}" if cnn_lbl != "Not run" else "—"
            mlp_sub  = f"{mlp_conf:.0%}" if mlp_lbl != "Not run" else "—"
            st.markdown(
                '<div class="tt-section">Classification</div>'
                f'<div class="tt-row">'
                f'<span class="tt-row-label">CNN</span>'
                f'<span><span class="tt-row-value">{cnn_lbl}</span>'
                f'<span class="tt-row-sub">{cnn_sub}</span></span></div>'
                f'<div class="tt-row">'
                f'<span class="tt-row-label">MLP</span>'
                f'<span><span class="tt-row-value">{mlp_lbl}</span>'
                f'<span class="tt-row-sub">{mlp_sub}</span></span></div>',
                unsafe_allow_html=True,
            )

            # TRAJECTORY
            hist_len = len(st.session_state.trajectory_history)
            pred     = st.session_state.trajectory_prediction
            pred_str = f"({pred[0]:.3f}, {pred[1]:.3f})" if pred else "—"
            st.markdown(
                '<div class="tt-section">Trajectory</div>'
                f'<div class="tt-row">'
                f'<span class="tt-row-label">Positions</span>'
                f'<span class="tt-row-value">{hist_len}</span></div>'
                f'<div class="tt-row">'
                f'<span class="tt-row-label">LSTM next</span>'
                f'<span class="tt-row-value" style="font-size:10px;font-weight:500">{pred_str}</span></div>',
                unsafe_allow_html=True,
            )

            # DISPLAY
            st.markdown('<div class="tt-section">Display</div>', unsafe_allow_html=True)
            show_boxes      = st.checkbox("Boxes",      value=True)
            show_labels     = st.checkbox("Labels",     value=True)
            show_confidence = st.checkbox("Confidence", value=True)
            show_keypoints  = st.checkbox("Keypoints",  value=False)

        with left_col:
            if uploaded_image is not None:
                out_img = image.copy()
                draw    = ImageDraw.Draw(out_img)
                if show_boxes:
                    names = {1: "Player", 2: "Ball"}
                    p_idx = 0
                    for box, score, label in zip(
                        st.session_state.detector_boxes,
                        st.session_state.detector_scores,
                        st.session_state.detector_labels,
                    ):
                        x1, y1, x2, y2 = [int(v) for v in box]
                        color = "red" if label == 1 else "#00CFFF"
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                        parts = []
                        if show_labels:     parts.append(names.get(label, "?"))
                        if show_confidence: parts.append(f"{score:.2f}")
                        if parts:
                            draw.text((x1 + 3, max(0, y1 - 16)),
                                      " ".join(parts), fill=color)
                        if label == 1:
                            p_idx += 1
                if show_keypoints:
                    kp_draw = ImageDraw.Draw(out_img, "RGBA")
                    for x, y in st.session_state.keypoints:
                        kp_draw.ellipse([x-5, y-5, x+5, y+5],
                                        fill=(0, 255, 220, 200), outline=(0, 0, 0, 200))
                st.image(out_img, use_container_width=True)
            else:
                st.markdown(
                    '<div class="tt-empty">'
                    '<div class="tt-empty-icon">'
                    '<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#555" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round">'
                    '<rect x="3" y="3" width="18" height="18" rx="2"/>'
                    '<circle cx="8.5" cy="8.5" r="1.5"/>'
                    '<polyline points="21 15 16 10 5 21"/>'
                    '</svg>'
                    '</div>'
                    '<div class="tt-empty-title">No image loaded</div>'
                    '<div class="tt-empty-sub">Upload an image from the sidebar to begin analysis</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

    # ── VIDEO MODE ────────────────────────────────────────────────────────
    elif input_mode == "Video":
        if uploaded_video is None:
            st.markdown(
                '<div class="tt-empty">'
                '<div class="tt-empty-icon">'
                '<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#555" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round">'
                '<polygon points="23 7 16 12 23 17 23 7"/>'
                '<rect x="1" y="5" width="15" height="14" rx="2"/>'
                '</svg>'
                '</div>'
                '<div class="tt-empty-title">No video loaded</div>'
                '<div class="tt-empty-sub">Upload a video from the sidebar to begin processing</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        elif not st.session_state.video_processed:
            if st.button("Process video", type="primary"):
                with st.spinner("Running detection on frames…"):
                    out_path, vstats = process_video(uploaded_video, confidence_threshold)
                st.session_state.video_output_path = out_path
                st.session_state.video_stats       = vstats
                st.session_state.video_processed   = True
                st.rerun()

        else:
            out_path = st.session_state.video_output_path
            vs       = st.session_state.video_stats
            n        = len(vs)

            if not out_path or not Path(out_path).exists():
                st.warning("Output file not found — click Re-process.")
            elif n == 0:
                st.warning("No detections — try lowering the confidence threshold.")
            else:
                with open(out_path, "rb") as vf:
                    st.video(vf.read())

                n_players_max = max((s["n_players"] for s in vs), default=0)
                ball_frames   = sum(1 for s in vs if s["n_balls"] > 0)
                lstm_frames   = sum(1 for s in vs if s["lstm_pred"] is not None)
                st.markdown(
                    f'<p style="font-size:11px;color:rgba(255,255,255,0.35);margin:6px 0 2px">'
                    f'{n} frames · {n_players_max} players max · '
                    f'ball {ball_frames}/{n} frames · LSTM {lstm_frames} frames'
                    f'</p>',
                    unsafe_allow_html=True,
                )

            if st.button("Re-process"):
                if out_path:
                    try: os.unlink(out_path)
                    except OSError: pass
                st.session_state.video_output_path = None
                st.session_state.video_stats       = []
                st.session_state.video_processed   = False
                st.rerun()

    # ── CAMERA MODE ───────────────────────────────────────────────────────
    elif input_mode == "Camera":
        left_col, right_col = st.columns([3, 1], gap="large")
        with right_col:
            st.markdown('<div class="tt-section" style="margin-top:0">Detections</div>', unsafe_allow_html=True)
            st.metric("Objects", st.session_state.player_count)

            cnn_lbl  = st.session_state.classifier_label
            cnn_conf = st.session_state.classifier_confidence
            mlp_lbl  = st.session_state.mlp_label
            mlp_conf = st.session_state.mlp_confidence
            cnn_sub  = f"{cnn_conf:.0%}" if cnn_lbl != "Not run" else "—"
            mlp_sub  = f"{mlp_conf:.0%}" if mlp_lbl != "Not run" else "—"
            st.markdown(
                '<div class="tt-section">Classification</div>'
                f'<div class="tt-row"><span class="tt-row-label">CNN</span>'
                f'<span><span class="tt-row-value">{cnn_lbl}</span>'
                f'<span class="tt-row-sub">{cnn_sub}</span></span></div>'
                f'<div class="tt-row"><span class="tt-row-label">MLP</span>'
                f'<span><span class="tt-row-value">{mlp_lbl}</span>'
                f'<span class="tt-row-sub">{mlp_sub}</span></span></div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="tt-section">Display</div>', unsafe_allow_html=True)
            show_boxes_c     = st.checkbox("Boxes",     value=True,  key="cb_c")
            show_keypoints_c = st.checkbox("Keypoints", value=False, key="ck_c")

        with left_col:
            camera_photo = st.camera_input("Capture frame")
            if camera_photo is not None:
                image = Image.open(io.BytesIO(camera_photo.getvalue())).convert("RGB")
                st.session_state.base_image = image
                if run_model:
                    with st.spinner("Running full analysis…"):
                        run_full_analysis(image, confidence_threshold)
                out_img = image.copy()
                draw    = ImageDraw.Draw(out_img, "RGBA")
                if show_boxes_c:
                    names={1:"Player",2:"Ball"}
                    for i,(box,score,label) in enumerate(zip(
                        st.session_state.detector_boxes,
                        st.session_state.detector_scores,
                        st.session_state.detector_labels,
                    )):
                        x1,y1,x2,y2 = box
                        c   = _PLAYER_COLORS[i] if label==1 else _BALL_COLOR
                        rgb = _hex_to_rgb(c)
                        L   = min(20,int(x2-x1)//4,int(y2-y1)//4)
                        for (ax,ay,bx,by,cx,cy) in [
                            (x1,y1,x1+L,y1,x1,y1+L),(x2,y1,x2-L,y1,x2,y1+L),
                            (x1,y2,x1+L,y2,x1,y2-L),(x2,y2,x2-L,y2,x2,y2-L),
                        ]:
                            draw.line([(ax,ay),(bx,by)], fill=rgb+(255,), width=3)
                            draw.line([(ax,ay),(cx,cy)], fill=rgb+(255,), width=3)
                        draw.text((x1+2,max(0,y1-16)),
                                  f"{names.get(label,'?')} {score:.2f}", fill=rgb+(255,))
                if show_keypoints_c:
                    for x,y in st.session_state.keypoints:
                        draw.ellipse([x-5,y-5,x+5,y+5], fill=(0,255,220,200))
                st.image(out_img, use_container_width=True)
            else:
                st.markdown(
                    '<div class="tt-empty" style="min-height:300px">'
                    '<div class="tt-empty-icon">'
                    '<svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#555" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round">'
                    '<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>'
                    '<circle cx="12" cy="13" r="4"/>'
                    '</svg>'
                    '</div>'
                    '<div class="tt-empty-title">Camera ready</div>'
                    '<div class="tt-empty-sub">Capture a frame above to run analysis</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────
# TAB 2 — 2D COURT VIEW
# ─────────────────────────────────────────────
with court_tab:
    img_w, img_h = (
        st.session_state.base_image.size if st.session_state.base_image else (0, 0)
    )
    court_bytes = build_court_image(
        st.session_state.trajectory_history,
        st.session_state.trajectory_prediction,
        st.session_state.detector_boxes,
        img_w, img_h,
    )

    hist_len = len(st.session_state.trajectory_history)

    court_col, info_col = st.columns([1, 1], gap="large")
    with court_col:
        st.image(court_bytes, use_container_width=True)

    with info_col:
        st.markdown('<div class="tt-section" style="margin-top:0">Court View</div>', unsafe_allow_html=True)
        if hist_len > 0:
            n_players = sum(1 for l in st.session_state.detector_labels if l == 1)
            n_balls   = sum(1 for l in st.session_state.detector_labels if l == 2)
            pred = st.session_state.trajectory_prediction
            pred_str = f"({pred[0]:.3f}, {pred[1]:.3f})" if pred else "—"
            st.markdown(
                f'<div class="tt-row"><span class="tt-row-label">Positions tracked</span>'
                f'<span class="tt-row-value">{hist_len}</span></div>'
                f'<div class="tt-row"><span class="tt-row-label">Players (last frame)</span>'
                f'<span class="tt-row-value">{n_players}</span></div>'
                f'<div class="tt-row"><span class="tt-row-label">Balls (last frame)</span>'
                f'<span class="tt-row-value">{n_balls}</span></div>'
                f'<div class="tt-row"><span class="tt-row-label">LSTM next</span>'
                f'<span class="tt-row-value" style="font-size:10px;font-weight:500">{pred_str}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size:10px;color:rgba(255,255,255,0.2);line-height:1.7">'
                'Dots fade from dim → bright (oldest → latest).<br>'
                'Colored dots show player/ball positions from the last frame.<br>'
                '★ marks the LSTM predicted next position.'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="tt-empty" style="min-height:260px">'
                '<div class="tt-empty-icon">'
                '<svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#555" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round">'
                '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>'
                '</svg>'
                '</div>'
                '<div class="tt-empty-title">No tracking data</div>'
                '<div class="tt-empty-sub">Run analysis on an image or process a video first</div>'
                '</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────
# TAB 3 — VLM CHAT
# ─────────────────────────────────────────────
with chat_tab:

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask something about the tennis scene…")
    if prompt:
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                # ── Pick the best available image for VLM context ──────
                image_file = None
                if input_mode == "Image" and uploaded_image:
                    uploaded_image.seek(0)
                    image_file = uploaded_image
                elif input_mode == "Video" and st.session_state.video_frame:
                    st.session_state.video_frame.seek(0)
                    image_file = st.session_state.video_frame
                elif input_mode == "Camera" and st.session_state.base_image:
                    _cam_buf = io.BytesIO()
                    st.session_state.base_image.save(_cam_buf, format="JPEG", quality=90)
                    _cam_buf.seek(0)
                    image_file = _cam_buf

                # ── Build detection context ────────────────────────────
                detection_ctx = None
                if input_mode == "Video" and st.session_state.video_stats:
                    vs = st.session_state.video_stats
                    n = len(vs)
                    n_players_max = max((s["n_players"] for s in vs), default=0)
                    ball_frames   = sum(1 for s in vs if s["n_balls"] > 0)
                    lstm_frames   = sum(1 for s in vs if s["lstm_pred"] is not None)
                    detection_ctx = (
                        f"Video analysis over {n} frames: "
                        f"up to {n_players_max} player(s) detected, "
                        f"ball tracked in {ball_frames}/{n} frames, "
                        f"LSTM trajectory prediction active for {lstm_frames} frames. "
                        f"The attached image is a representative mid-video frame."
                    )
                elif st.session_state.player_count > 0:
                    boxes_str = "; ".join(
                        f"box{i+1}=[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}] score={s:.2f}"
                        for i,(b,s) in enumerate(zip(
                            st.session_state.detector_boxes,
                            st.session_state.detector_scores,
                        ))
                    )
                    detection_ctx = (
                        f"{st.session_state.player_count} object(s) detected: {boxes_str}. "
                        f"CNN={st.session_state.classifier_label}"
                        f"({st.session_state.classifier_confidence:.2f}), "
                        f"MLP={st.session_state.mlp_label}"
                        f"({st.session_state.mlp_confidence:.2f})."
                    )
                    if st.session_state.trajectory_prediction:
                        px, py = st.session_state.trajectory_prediction
                        detection_ctx += f" LSTM predicts next position ({px:.3f},{py:.3f})."

                try:
                    response_text = ask_vlm(prompt, image_file,
                                            detection_context=detection_ctx)
                    add_log("VLM prompt sent successfully")
                except Exception as e:
                    response_text = f"VLM error: {e}"
                    add_log(f"VLM error: {e}", "ERROR")
            st.markdown(response_text)
        st.session_state.messages.append({"role":"assistant","content":response_text})


# ─────────────────────────────────────────────
# TAB 4 — TRAINING
# ─────────────────────────────────────────────
with training_tab:
    st.markdown('<div class="tt-section" style="margin-top:0">Train a Module</div>', unsafe_allow_html=True)
    module_choice = st.selectbox("Module", [
        "MLP",
        "CNN Classifier",
        "CNN Keypoints",
        "LSTM",
        "Detector (Faster R-CNN)",
    ])
    defaults_map = {
        "MLP":                    (40,  1e-3,  128),
        "CNN Classifier":         (20,  1e-3,   32),
        "CNN Keypoints":          (30,  1e-3,    8),
        "LSTM":                   (50,  1e-3,   64),
        "Detector (Faster R-CNN)":(20,  5e-3,    4),
    }
    def_e, def_l, def_b = defaults_map[module_choice]
    col_e, col_l, col_b = st.columns(3)
    with col_e: t_epochs = st.number_input("Epochs",     1,   200, def_e)
    with col_l: t_lr     = st.number_input("LR",    1e-5, 0.1, def_l, format="%.5f")
    with col_b: t_batch  = st.number_input("Batch",      1,   512, def_b)

    if st.button("Train", type="primary"):
        add_log(f"Starting: {module_choice} | epochs={t_epochs} lr={t_lr} batch={t_batch}")
        with st.spinner(f"Training {module_choice}…"):
            if module_choice == "MLP":
                from modules.mlp.train import train as _train
                out = _capture_and_train(_train, epochs=int(t_epochs),
                                         lr=float(t_lr), batch_size=int(t_batch))
            elif module_choice == "CNN Classifier":
                from modules.cnn.train import train_classifier as _train
                out = _capture_and_train(_train, epochs=int(t_epochs),
                                         lr=float(t_lr), batch_size=int(t_batch))
            elif module_choice == "CNN Keypoints":
                from modules.cnn.train import train_keypoint_detector as _train
                out = _capture_and_train(_train, epochs=int(t_epochs),
                                         lr=float(t_lr), batch_size=int(t_batch))
            elif module_choice == "LSTM":
                from modules.lstm.train import train as _train
                out = _capture_and_train(_train, epochs=int(t_epochs),
                                         lr=float(t_lr), batch_size=int(t_batch))
            elif module_choice == "Detector (Faster R-CNN)":
                from modules.detector.train import train as _train
                out = _capture_and_train(_train, epochs=int(t_epochs),
                                         lr=float(t_lr), batch_size=int(t_batch))
        for line in out.strip().splitlines():
            add_log(line)
        st.success("Done — weights saved. Check Logs tab.")
        st.cache_resource.clear()
        st.rerun()



# ─────────────────────────────────────────────
# TAB 5 — SYSTEM LOGS
# ─────────────────────────────────────────────
with logs_tab:
    st.markdown('<div class="tt-section" style="margin-top:0">System Logs</div>', unsafe_allow_html=True)
    if st.button("Clear Logs"):
        st.session_state.logs = []
        add_log("Logs cleared")
    if st.session_state.logs:
        st.code("\n".join(st.session_state.logs[-200:]), language=None)
    else:
        st.markdown(
            '<div style="color:rgba(255,255,255,0.2);font-size:12px;padding:24px 0">No log entries yet.</div>',
            unsafe_allow_html=True,
        )

