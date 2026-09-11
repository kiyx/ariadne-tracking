"""Costanti e parametri centralizzati della pipeline Ariadne Tracking.

Tutti i valori sono sovrascrivibili dai flag CLI dei singoli moduli.
"""

from __future__ import annotations

from pathlib import Path

# ── Path del progetto ─────────────────────────────────────────

_SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SRC_DIR.parent

# --- Detection / tracking ---

PERSON_CLASS_ID = 0
MIN_WIDTH = 25
MIN_HEIGHT = 75
MAX_ASPECT_RATIO = 0.75
MIN_ASPECT_RATIO = 0.2
DEFAULT_CONF = 0.5
DEFAULT_FRAME_SKIP = 3  # salva 1 frame ogni 3
DEFAULT_IMGSZ = 640
DEFAULT_TRACKER = "botsort.yaml"

# --- Estrazione ROI ---

ROI_RESIZE = (128, 256)  # ingresso standard per la maggior parte dei modelli Re-ID
PADDING_RATIO = 0.05
JPEG_QUALITY = 95

# --- Filtri sovrapposizione (stile MEVID paper) ---

OVERLAP_IOU_THRESHOLD = 0.3
CONTAINMENT_THRESHOLD = 0.5

# --- Detection parziali sul bordo ---

EDGE_MARGIN_RATIO = 0.02

# --- Pose-guided filtering ---

# Sotto 150 px i keypoint COCO sono troppo instabili.
POSE_KPT_MIN_HEIGHT = 150
POSE_KPT_CONF_THRESHOLD = 0.5
# Upper-body COCO: nose, eyes, ears, shoulders. Se ne mancano troppi, la detection
# è probabilmente tagliata (gambe/piedi) e non serve per Re-ID.
POSE_MIN_UPPER_KEYPOINTS = 2

# ── Formati video supportati ──────────────────────────────────

VIDEO_EXTENSIONS = frozenset((".mp4", ".avi", ".mov", ".mkv", ".webm"))

# ── Path di default (relativi a PROJECT_ROOT) ─────────────────

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "videos"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "extracted_rois"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "yolo26m-pose.pt"
LOG_DIR = PROJECT_ROOT / "output" / "logs"

# --- Qualità ROI ---

# Calibrata sul subset MEVID: sotto ~10 = sfocata, sopra ~20 = nitida.
SHARPNESS_THRESHOLD = 15

# --- Filtro track ---

MIN_TRACK_FRAMES = 8
# Media della similarità coseno tra le clip di una stessa tracklet.
# Se troppo bassa, le clip probabilmente contengono occlusioni o cambi di pose.
MIN_INTRA_TRACK_SIMILARITY = 0.4

# --- ReID / Feature extraction ---

SEQ_LEN = 8  # atteso da C2DResNet50 (protocollo CCVID)
DEFAULT_SAMPLING_STRIDE = 4  # campionamento denso tra clip sovrapposte
DEFAULT_REID_WEIGHTS = PROJECT_ROOT / "models" / "CAL_best_model.pth.tar"

# --- Seed ---

DEFAULT_SEED = 67

# --- Clustering gerarchico (Modulo 3) ---

# Penalità cannot-link: supera il massimo teorico 2.0 della distanza coseno,
# così due tracklet simultanee dello stesso video non finiscono mai insieme.
CANNOT_LINK_PENALTY = 10.0
