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
MIN_WIDTH = 25
MIN_HEIGHT = 75
MAX_ASPECT_RATIO = 0.75
MIN_ASPECT_RATIO = 0.2
DEFAULT_CONF = 0.5
DEFAULT_FRAME_SKIP = 3
DEFAULT_IMGSZ = 640
DEFAULT_TRACKER = "botsort.yaml"

# ── Costanti di estrazione ROI ────────────────────────────────

ROI_RESIZE = (128, 256)  # (w, h)
PADDING_RATIO = 0.05
JPEG_QUALITY = 95

# ── Filtri sovrapposizione (stile MEVID paper) ────────────────

OVERLAP_IOU_THRESHOLD = 0.3  # Detection con IoU > 0.3 → scarta la più piccola
CONTAINMENT_THRESHOLD = 0.5  # Sopprime bbox contenute per >=50% in una più grande

# ── Filtro bordo frame (detection parziali) ───────────────────

EDGE_MARGIN_RATIO = 0.02  # Se la bbox tocca il bordo entro il 2% del frame → partial

# ── Pose-guided filtering ─────────────────────────────────────

# Altezza minima della bbox (in pixel) per fidarsi dei keypoint.
# Sotto questa soglia i keypoint sono inaffidabili e si usa il fallback
# euristico (is_edge_bbox). Sopra, il filtro usa i keypoint COCO.
POSE_KPT_MIN_HEIGHT = 150
# Confidenza minima perché un keypoint sia considerato "visibile"
POSE_KPT_CONF_THRESHOLD = 0.5
# Numero minimo di keypoint upper-body (indici 0-6: nose, occhi, orecchie,
# spalle) che devono essere visibili per considerare la detection completa.
# Se meno di questo → detection parziale (solo gambe/piedi/torso basso).
POSE_MIN_UPPER_KEYPOINTS = 2

# ── Formati video supportati ──────────────────────────────────

VIDEO_EXTENSIONS = frozenset((".mp4", ".avi", ".mov", ".mkv", ".webm"))

# ── Path di default (relativi a PROJECT_ROOT) ─────────────────

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "videos"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "extracted_rois"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "yolo26m-pose.pt"
LOG_DIR = PROJECT_ROOT / "output" / "logs"

# ── Soglia qualità ROI ─────────────────────────────────────────

SHARPNESS_THRESHOLD = 15

# ── Filtro track ────────────────────────────────────────────────

MIN_TRACK_FRAMES = 8  # Track con meno di N frame vengono scartate
MIN_INTRA_TRACK_SIMILARITY = 0.4  # Similarità coseno minima intra-track

# ── ReID / Feature extraction ─────────────────────────────────

SEQ_LEN = 8  # Lunghezza sequenza temporale per C2DResNet50
DEFAULT_SAMPLING_STRIDE = 4  # Stride temporale sampling clip (protocollo CCVID)
DEFAULT_REID_WEIGHTS = PROJECT_ROOT / "models" / "CAL_best_model.pth.tar"

# ── Seed di default ───────────────────────────────────────────

DEFAULT_SEED = 67
