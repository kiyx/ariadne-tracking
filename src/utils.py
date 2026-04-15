"""Funzioni helper riusabili per la pipeline Ariadne Tracking.

Raccoglie utility pure (non side-effect) per bounding-box,
validazione ROI, path dei modelli TensorRT, metadati video,
caricamento embedding e raggruppamento video per scena.
Tutte le funzioni sono testabili in isolamento.
"""

from __future__ import annotations

import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torchvision import transforms

from src.config import (
    CONTAINMENT_THRESHOLD,
    LOG_DIR,
    MAX_ASPECT_RATIO,
    MIN_ASPECT_RATIO,
    MIN_HEIGHT,
    MIN_WIDTH,
    PADDING_RATIO,
)

# ── Rich (opzionale) ─────────────────────────────────────────

try:
    from rich.logging import RichHandler

    _RICH = True
except ImportError:
    _RICH = False

# ── Dataclass condivise ───────────────────────────────────────


@dataclass
class VideoInfo:
    """Metadati base di un file video (risoluzione, fps, durata)."""

    frame_w: int
    frame_h: int
    fps: float
    total_frames: int


def pad_and_clip_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    frame_w: int,
    frame_h: int,
    ratio: float = PADDING_RATIO,
) -> tuple[int, int, int, int]:
    """
    Espande la bounding box proporzionalmente e la clippa ai bordi del frame.

    Il padding è proporzionale (non fisso in pixel) così che bbox grandi
    ricevano un margine maggiore e bbox piccole uno adeguato alla loro scala.
    """
    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = int(w * ratio), int(h * ratio)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(frame_w, x2 + pad_x),
        min(frame_h, y2 + pad_y),
    )


def is_valid_roi(w: int, h: int) -> bool:
    """
    Verifica se una ROI è utilizzabile per la Re-Identification.

    Criteri:
      1. Dimensione minima — crop troppo piccoli non hanno abbastanza dettaglio
      2. Aspect ratio massimo — w/h > MAX_ASPECT_RATIO indica un falso positivo
      3. Aspect ratio minimo — w/h < MIN_ASPECT_RATIO indica un artefatto (palo, bordo)
    """
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return False
    ratio = w / h if h > 0 else float("inf")
    return MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO


def suppress_contained_boxes(
    boxes: np.ndarray,
    threshold: float = CONTAINMENT_THRESHOLD,
) -> np.ndarray:
    """Sopprime bbox più piccole contenute in bbox più grandi.

    Per ogni coppia di detection, se l'area di intersezione copre
    >= ``threshold`` della bbox più piccola, quest'ultima viene scartata.
    Risolve le detection duplicate di piedi/parti del corpo che YOLO
    produce quando una persona è già rilevata a corpo intero.

    Complessità: O(N²) ma completamente vettorizzato con NumPy
    (N = detection per frame, tipicamente 5–30).

    Args:
        boxes: Array (N, 4) in formato xyxy.
        threshold: Frazione minima di contenimento per sopprimere.

    Returns:
        Indici booleani (N,) — True = mantieni, False = sopprimi.
    """
    n = len(boxes)
    if n <= 1:
        return np.ones(n, dtype=bool)

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Intersezione vettorizzata: broadcasting (N, 1, 4) vs (1, N, 4)
    x1 = np.maximum(boxes[:, 0:1], boxes[:, 0])  # (N, N)
    y1 = np.maximum(boxes[:, 1:2], boxes[:, 1])
    x2 = np.minimum(boxes[:, 2:3], boxes[:, 2])
    y2 = np.minimum(boxes[:, 3:4], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)  # (N, N)

    # Per ogni coppia, la bbox con area minore
    min_area = np.minimum(areas[:, None], areas[None, :])  # (N, N)

    # Rapporto di contenimento (evita divisione per zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(min_area > 0, inter / min_area, 0.0)

    # Azzera diagonale (una box non sopprime sé stessa)
    np.fill_diagonal(ratio, 0.0)

    # Una bbox è soppressa se è la più piccola nella coppia E ratio >= threshold
    is_smaller = areas[:, None] <= areas[None, :]  # (i, j): True se i è più piccola di j
    suppressed = np.any((ratio >= threshold) & is_smaller, axis=1)

    return cast("np.ndarray", ~suppressed)


def get_engine_path(pt_path: str, imgsz: int) -> Path:
    """
    Costruisce il path del file .engine TensorRT a partire dal .pt originale.

    Convenzione: <nome_modello>_fp16_<imgsz>.engine nella stessa directory.
    Codificare imgsz nel nome evita conflitti tra motori compilati con
    risoluzioni diverse (640, 1280, ecc.).
    """
    p = Path(pt_path)
    return p.parent / f"{p.stem}_fp16_{imgsz}.engine"


def roi_sharpness(roi: np.ndarray) -> float:
    """Calcola la nitidezza di una ROI usando la varianza del Laplaciano.

    Un valore alto indica un'immagine nitida; un valore basso indica
    sfocatura o occlusione parziale.

    Args:
        roi: Immagine BGR (numpy array) della ROI.

    Returns:
        Varianza del Laplaciano (float). Tipicamente:
        - < 50  → molto sfocata
        - 50–150 → mediocre
        - > 150 → buona
    """
    import cv2

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def get_video_info(video_path: str) -> VideoInfo | None:
    """Estrae metadati del video (risoluzione, fps, frame count).

    Apre e rilascia subito l'handle per evitare conflitti con
    ``model.track()`` su Windows (due handle sullo stesso file = lock).
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    info = VideoInfo(
        frame_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        frame_h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 30.0),
        total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return info


# ══════════════════════════════════════════════════════════════
# Logging condiviso (console + file)
# ══════════════════════════════════════════════════════════════


def setup_logging(module_name: str = "run") -> Path | None:
    """Configura logging su console (Rich o plain) e su file.

    Il file viene creato in ``output/logs/<module_name>_<timestamp>.log``.
    Il guard ``if root.handlers`` evita handler duplicati se la funzione
    viene chiamata più volte (ad es. nei test).
    """
    root = logging.getLogger()
    if root.handlers:
        return None
    root.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    # Console handler: Rich (colori) se disponibile, altrimenti plain
    if _RICH:
        ch: logging.Handler = RichHandler(
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        ch.setFormatter(logging.Formatter("%(message)s"))
    else:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
    root.addHandler(ch)

    # File handler
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOG_DIR / f"{module_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
        return log_file
    except OSError:
        return None


# ══════════════════════════════════════════════════════════════
# ReID: trasformazione e caricamento modello
# ══════════════════════════════════════════════════════════════

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((256, 128), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_reid_model(weights_path: Path, device: torch.device) -> torch.nn.Module:
    """Carica C2DResNet50 con pesi CAL pre-addestrati.

    Gestisce la pulizia delle chiavi del dizionario di stato (rimuove il
    prefisso ``module.`` generato da DataParallel) e valida il numero
    di chiavi caricate.
    """
    from models.simple_ccreid.configs.default_vid import _C  # pyright: ignore
    from models.simple_ccreid.models.vid_resnet import C2DResNet50  # pyright: ignore

    log = logging.getLogger(__name__)

    config = _C.clone()
    model = C2DResNet50(config)

    if not weights_path.exists():
        raise FileNotFoundError(f"Pesi CAL non trovati in: {weights_path}")

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        log.warning("Chiavi mancanti nel checkpoint: %s", load_result.missing_keys)
    if load_result.unexpected_keys:
        log.warning("Chiavi inattese nel checkpoint: %s", load_result.unexpected_keys)

    n_model = len(model.state_dict())
    n_loaded = n_model - len(load_result.missing_keys)
    log.info("Pesi caricati: %d/%d chiavi dal checkpoint.", n_loaded, n_model)
    if n_loaded < n_model * 0.5:
        raise RuntimeError(
            f"Solo {n_loaded}/{n_model} chiavi caricate. "
            "Possibile mismatch checkpoint/architettura."
        )

    model.to(device)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════
# Embedding & raggruppamento video per scena
# ══════════════════════════════════════════════════════════════

# Pattern per estrarre scena e camera ID dal nome video MEVID.
# Es: "2018-03-11.14-05-01.14-10-01.school.G328.r13"
#   → scene = "2018-03-11.14-05-01.14-10-01.school"
#   → cam   = "G328"
_VIDEO_PATTERN = re.compile(r"^(.+)\.(G\d+)\.r\d+$")

# Pattern più specifico per raggruppare video con offset di secondi diversi.
# Cattura: data, ora_inizio(HH-MM), ora_fine(HH-MM), location — ignorando i secondi.
_SCENE_PATTERN = re.compile(
    r"^(\d{4}-\d{2}-\d{2})\.(\d{2}-\d{2})-\d{2}\.(\d{2}-\d{2})-\d{2}\.(\w+)\.(G\d+)\.r\d+$"
)


def load_all_embeddings(input_dir: Path) -> dict[str, dict[int, torch.Tensor]]:
    """Carica ``embeddings.pt`` da tutte le directory video.

    Returns:
        Dict ``{video_name: {track_id: embedding_tensor}}``.
    """
    log = logging.getLogger(__name__)
    result: dict[str, dict[int, torch.Tensor]] = {}
    for emb_file in sorted(input_dir.glob("*/embeddings.pt")):
        video_name = emb_file.parent.name
        data = torch.load(emb_file, map_location="cpu", weights_only=False)
        if data:
            result[video_name] = data
            log.info("Caricato %s: %d identità", video_name, len(data))
    return result


def group_by_scene(video_names: list[str]) -> dict[str, list[str]]:
    """Raggruppa i video per scena (stessa data + location, camera diversa).

    I video con nome non corrispondente al pattern MEVID vengono
    raggruppati sotto ``_ungrouped``.
    """
    scenes: dict[str, list[str]] = defaultdict(list)
    for name in video_names:
        m = _SCENE_PATTERN.match(name)
        if m:
            date, location = m.group(1), m.group(4)
            scene_key = f"{date}.{location}"
            scenes[scene_key].append(name)
        else:
            m2 = _VIDEO_PATTERN.match(name)
            if m2:
                scenes[m2.group(1)].append(name)
            else:
                scenes["_ungrouped"].append(name)
    return dict(scenes)


def get_camera_id(video_name: str) -> str:
    """Estrae l'ID della telecamera dal nome video (es. 'G328')."""
    m = _VIDEO_PATTERN.match(video_name)
    return m.group(2) if m else video_name
