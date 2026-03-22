"""Configurazione centralizzata del progetto Ariadne Tracking.

Questo modulo definisce tutte le costanti e i parametri della pipeline
in un unico punto. I valori possono essere sovrascritti dai flag CLI
dei singoli moduli.
"""

from __future__ import annotations

from pathlib import Path

# ── Path del progetto ─────────────────────────────────────────

_SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SRC_DIR.parent

# ── Costanti di detection / tracking ──────────────────────────

PERSON_CLASS_ID = 0
MIN_WIDTH = 32
MIN_HEIGHT = 64
MAX_ASPECT_RATIO = 1.5
DEFAULT_CONF = 0.50
DEFAULT_FRAME_SKIP = 3
DEFAULT_IMGSZ = 640
DEFAULT_TRACKER = "botsort.yaml"

# ── Costanti di estrazione ROI ────────────────────────────────

ROI_RESIZE = (128, 256)  # (w, h)
PADDING_RATIO = 0.05
JPEG_QUALITY = 95

# ── Formati video supportati ──────────────────────────────────

VIDEO_EXTENSIONS = frozenset((".mp4", ".avi", ".mov", ".mkv", ".webm"))

# ── Path di default (relativi a PROJECT_ROOT) ─────────────────

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "videos"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "extracted_rois"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "yolo26m.pt"
LOG_DIR = PROJECT_ROOT / "output" / "logs"

# ── Soglia qualità ROI ─────────────────────────────────────────

SHARPNESS_THRESHOLD = 100

# ── Seed di default ───────────────────────────────────────────

DEFAULT_SEED = 67
