"""Configurazione centralizzata del progetto Ariadne Tracking.

Questo modulo definisce tutte le costanti e i parametri della pipeline
in un unico punto. La catena di priorità è:

    hardcoded defaults  →  configs/default.yaml  →  CLI flags

All'importazione, il file YAML viene caricato automaticamente e i valori
presenti sovrascrivono i default corrispondenti.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

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
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "yolo26n.pt"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
LOG_DIR = PROJECT_ROOT / "output" / "logs"

# ── Seed di default ───────────────────────────────────────────

DEFAULT_SEED = 67


def load_yaml_config(path: Path | None = None) -> dict[str, Any]:
    """
    Carica un file YAML di configurazione.

    Se il file non esiste, restituisce un dizionario vuoto
    (i default hardcoded sopra verranno usati).
    """
    if path is None:
        path = DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


# ── Override dai valori YAML (se il file esiste) ─────────────

_yaml_cfg = load_yaml_config()

if _yaml_cfg:
    # Detection / tracking
    DEFAULT_CONF = _yaml_cfg.get("conf", DEFAULT_CONF)
    DEFAULT_FRAME_SKIP = _yaml_cfg.get("frame_skip", DEFAULT_FRAME_SKIP)
    DEFAULT_IMGSZ = _yaml_cfg.get("imgsz", DEFAULT_IMGSZ)
    DEFAULT_TRACKER = _yaml_cfg.get("tracker", DEFAULT_TRACKER)
    MIN_WIDTH = _yaml_cfg.get("min_width", MIN_WIDTH)
    MIN_HEIGHT = _yaml_cfg.get("min_height", MIN_HEIGHT)
    MAX_ASPECT_RATIO = _yaml_cfg.get("max_aspect_ratio", MAX_ASPECT_RATIO)

    # ROI extraction
    _roi = _yaml_cfg.get("roi_resize")
    if _roi and isinstance(_roi, list) and len(_roi) == 2:
        ROI_RESIZE = tuple(_roi)
    PADDING_RATIO = _yaml_cfg.get("padding_ratio", PADDING_RATIO)
    JPEG_QUALITY = _yaml_cfg.get("jpeg_quality", JPEG_QUALITY)

    # Paths (relativi a PROJECT_ROOT)
    if "model" in _yaml_cfg:
        DEFAULT_MODEL_PATH = PROJECT_ROOT / str(_yaml_cfg["model"])
    if "input_dir" in _yaml_cfg:
        DEFAULT_INPUT_DIR = PROJECT_ROOT / str(_yaml_cfg["input_dir"])
    if "output_dir" in _yaml_cfg:
        DEFAULT_OUTPUT_DIR = PROJECT_ROOT / str(_yaml_cfg["output_dir"])

    # Seed
    DEFAULT_SEED = _yaml_cfg.get("seed", DEFAULT_SEED)
