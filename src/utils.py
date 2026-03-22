"""Funzioni helper riusabili per la pipeline Ariadne Tracking.

Raccoglie utility pure (non side-effect) per bounding-box,
validazione ROI, path dei modelli TensorRT e metadati video.
Tutte le funzioni sono testabili in isolamento.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from config import (
    MAX_ASPECT_RATIO,
    MIN_HEIGHT,
    MIN_WIDTH,
    PADDING_RATIO,
)

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
      2. Aspect ratio — un pedone ha h > w; w/h > MAX_ASPECT_RATIO indica un falso positivo
    """
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return False
    return w <= h * MAX_ASPECT_RATIO


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
