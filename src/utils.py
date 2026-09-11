"""Helper puri usati in tutta la pipeline Ariadne Tracking.

Qui ci sono solo funzioni senza side-effect: bbox, validazione ROI,
metadati video, caricamento embedding, parsing dei nomi MEVID.
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
    EDGE_MARGIN_RATIO,
    LOG_DIR,
    MAX_ASPECT_RATIO,
    MIN_ASPECT_RATIO,
    MIN_HEIGHT,
    MIN_WIDTH,
    OVERLAP_IOU_THRESHOLD,
    PADDING_RATIO,
    POSE_KPT_CONF_THRESHOLD,
    POSE_KPT_MIN_HEIGHT,
    POSE_MIN_UPPER_KEYPOINTS,
)

# Rich opzionale per log colorato

try:
    from rich.logging import RichHandler

    _RICH = True
except ImportError:
    _RICH = False

# --- Dataclass condivise ---


@dataclass
class VideoInfo:
    """Risoluzione, fps e numero di frame di un video."""

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
    """Espande la bbox del fattore PADDING_RATIO e la clippa al frame.

    Padding proporzionale: mantiene lo stesso margine relativo indipendentemente
    dalla dimensione della persona.
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
    """Controlla se una ROI ha dimensioni e aspect ratio ragionevoli per Re-ID.

    I limiti vengono dal paper MEVID: altezza minima 75 px, larghezza minima 25 px.
    Aspect ratio troppo alto = crop parziale; troppo basso = artefatto.
    """
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return False
    ratio = w / h if h > 0 else float("inf")
    return MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO


def is_edge_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    frame_w: int,
    frame_h: int,
    margin_ratio: float = EDGE_MARGIN_RATIO,
) -> bool:
    """True se la bbox sembra una persona tagliata dal bordo inferiore.

    Filtriamo solo il bordo basso: i piedi tagliati sono un classico segnale
    di persona che sta entrando o uscendo dal frame. I bordi alto e laterali
    invece sono comuni e spesso comunque utili.
    """
    margin_y = int(frame_h * margin_ratio)
    at_bottom = y2 >= frame_h - margin_y
    at_top = y1 <= margin_y
    return at_bottom and not at_top


def is_partial_body(
    box: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    kpt_conf: np.ndarray | None = None,
) -> bool:
    """Scarta detection parziali usando keypoint o, in fallback, la geometria.

    Se la bbox è abbastanza alta (>= POSE_KPT_MIN_HEIGHT) ci fidiamo dei keypoint
    COCO: servono almeno POSE_MIN_UPPER_KEYPOINTS upper-body visibili, altrimenti
    la detection è probabilmente solo gambe/piedi.
    Se la bbox è piccola i keypoint sono instabili, quindi usiamo is_edge_bbox.
    """
    x1, y1, x2, y2 = box
    box_h = y2 - y1

    # Se abbiamo keypoint e la bbox è abbastanza grande da fidarsi
    if kpt_conf is not None and box_h >= POSE_KPT_MIN_HEIGHT:
        # Indici COCO upper-body: 0=nose, 1-2=eyes, 3-4=ears, 5-6=shoulders
        upper_visible = int((kpt_conf[:7] >= POSE_KPT_CONF_THRESHOLD).sum())
        return upper_visible < POSE_MIN_UPPER_KEYPOINTS

    # Fallback euristico per detection piccole o senza keypoint
    return is_edge_bbox(x1, y1, x2, y2, frame_w, frame_h)


def suppress_contained_boxes(
    boxes: np.ndarray,
    threshold: float = CONTAINMENT_THRESHOLD,
) -> np.ndarray:
    """Toglie bbox piccole contenute dentro bbox più grandi.

    YOLO a volte rileva piedi o parti del corpo dentro una detection completa:
    se una bbox è contenuta per almeno ``threshold`` in una più grande, la scartiamo.
    Vettorizzato in NumPy; N è piccolo, quindi O(N²) è accettabile.
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

    # Sopprime la bbox più piccola della coppia: < evita la soppressione
    # reciproca a parità di area.
    is_smaller = (
        areas[:, None] < areas[None, :]
    )  # (i, j): True se i è strettamente più piccola di j
    suppressed = np.any((ratio >= threshold) & is_smaller, axis=1)

    return cast("np.ndarray", ~suppressed)


def suppress_overlapping_boxes(
    boxes: np.ndarray,
    threshold: float = OVERLAP_IOU_THRESHOLD,
) -> np.ndarray:
    """Toglie detection sovrapposte con IoU alta (stile MEVID).

    Nelle scene affollate il tracker può confondersi: se due bbox si sovrappongono
    troppo, teniamo solo la più grande. Rispetto a suppress_contained_boxes qui
    basta una sovrapposizione parziale forte, non il contenimento completo.
    """
    n = len(boxes)
    if n <= 1:
        return np.ones(n, dtype=bool)

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    x1 = np.maximum(boxes[:, 0:1], boxes[:, 0])
    y1 = np.maximum(boxes[:, 1:2], boxes[:, 1])
    x2 = np.minimum(boxes[:, 2:3], boxes[:, 2])
    y2 = np.minimum(boxes[:, 3:4], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    union = areas[:, None] + areas[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)

    np.fill_diagonal(iou, 0.0)

    # Tie-breaker: a parità di area nessuna delle due viene soppressa.
    is_smaller = areas[:, None] < areas[None, :]
    suppressed = np.any((iou >= threshold) & is_smaller, axis=1)

    return cast("np.ndarray", ~suppressed)


def get_engine_path(pt_path: str, imgsz: int) -> Path:
    """Path dell'engine TensorRT derivato dal file .pt.

    Include imgsz nel nome per non mischiare engine compilati a risoluzioni diverse.
    """
    p = Path(pt_path)
    return p.parent / f"{p.stem}_fp16_{imgsz}.engine"


def roi_sharpness(roi: np.ndarray) -> float:
    """Varianza del Laplaciano della ROI: valore basso = sfocata.

    Misurata su ROI ridimensionate a 128x256; la soglia minima è
    SHARPNESS_THRESHOLD in config.
    """
    import cv2

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def get_video_info(video_path: str) -> VideoInfo | None:
    """Legge risoluzione, fps e numero di frame da un video.

    Apre e chiude subito l'handle: su Windows tenerne due aperti sullo stesso
    file crea lock che possono interferire con model.track().
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


# --- Logging condiviso ---


def setup_logging(module_name: str = "run") -> Path | None:
    """Attiva logging su console e su file.

    Scrive in output/logs/<module_name>_<timestamp>.log. Il controllo su
    root.handlers evita handler duplicati se chiamata più volte (es. nei test).
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


# --- ReID: trasformazione e caricamento modello ---

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((256, 128), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_reid_model(weights_path: Path, device: torch.device) -> torch.nn.Module:
    """Carica C2DResNet50 con pesi CAL.

    Rimuove il prefisso 'module.' lasciato da DataParallel e controlla che
    il numero di chiavi caricate sia sensato, altrimenti solleva un errore.
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


# --- Embedding e parsing metadati video ---

# Estrae scene e camera ID dai nomi video MEVID, es. ...school.G328.r13
_VIDEO_PATTERN = re.compile(r"^(.+)\.(G\d+)\.r\d+$")

# Come sopra ma ignora i secondi, così video con piccoli offset si raggruppano nella stessa scena.
_SCENE_PATTERN = re.compile(
    r"^(\d{4}-\d{2}-\d{2})\.(\d{2}-\d{2})-\d{2}\.(\d{2}-\d{2})-\d{2}\.(\w+)\.(G\d+)\.r\d+$"
)


def load_all_embeddings(input_dir: Path) -> dict[str, dict[int, torch.Tensor]]:
    """Carica embeddings.pt da ogni directory video.

    Ritorna {video_name: {track_id: embedding}}.
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
    """Raggruppa video che appartengono alla stessa scena (data + location).

    I nomi che non rispettano il pattern finiscono in _ungrouped.
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
    """Torna la camera ID dal nome video MEVID."""
    m = _VIDEO_PATTERN.match(video_name)
    return m.group(2) if m else video_name


# --- ReID: sampling e aggregazione embedding ---


@dataclass
class VideoMetadata:
    """Data, location e camera ID estratti dal nome video MEVID."""

    camera_id: str
    date: str
    scene_key: str
    camera_node_id: str
    location: str


def parse_video_metadata(video_name: str) -> VideoMetadata | None:
    """Torna data, location e camera ID dal nome video MEVID.

    Format: YYYY-MM-DD.HH-MM-SS.HH-MM-SS.location.G###.rN
    """
    # Prova pattern completo per data + location
    m = _SCENE_PATTERN.match(video_name)
    if m:
        date = m.group(1)
        location = m.group(4)
        camera_id = m.group(5)
        scene_key = f"{date}.{location}"
        camera_node_id = f"{camera_id}_{date}"
        return VideoMetadata(
            camera_id=camera_id,
            date=date,
            scene_key=scene_key,
            camera_node_id=camera_node_id,
            location=location,
        )

    # Fallback al pattern video generico
    m = _VIDEO_PATTERN.match(video_name)
    if not m:
        return None

    scene = m.group(1)
    camera_id = m.group(2)

    # Tenta di estrarre data e location dallo scene
    parts = scene.split(".")
    if len(parts) >= 4:
        date = parts[0]
        location = parts[3] if len(parts) > 3 else "unknown"
        scene_key = f"{date}.{location}"
        camera_node_id = f"{camera_id}_{date}"
    elif len(parts) >= 1:
        date = parts[0]
        location = "unknown"
        scene_key = f"{date}.unknown"
        camera_node_id = f"{camera_id}_{date}"
    else:
        date = "unknown"
        location = "unknown"
        scene_key = scene
        camera_node_id = camera_id

    return VideoMetadata(
        camera_id=camera_id,
        date=date,
        scene_key=scene_key,
        camera_node_id=camera_node_id,
        location=location,
    )


def get_video_absolute_start(video_name: str) -> float:
    """Timestamp epoch di inizio video, ricavato dal nome MEVID.

    Se il formato non è riconoscibile torna 0.0: è sufficiente per calcolare
    differenze temporali relative, anche se non assolute.
    """
    parts = video_name.split(".")
    if len(parts) >= 2:
        date_str = parts[0]
        time_str = parts[1].replace("-", ":")
        dt_str = f"{date_str} {time_str}"
        try:
            dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            return dt.timestamp()
        except ValueError:
            pass

    # Fallback: prova a estrarre solo la data
    if parts:
        try:
            dt = datetime.strptime(parts[0], "%Y-%m-%d")
            return dt.timestamp()
        except ValueError:
            pass

    return 0.0


def recombine_tracklet_clips(
    n_frames: int,
    seq_len: int,
    stride: int = 4,
) -> list[list[int]]:
    """Genera clip sovrapposte da una tracklet (protocollo CCVID).

    Copre tutta la sequenza con stride fisso, come fa simple_ccreid.
    """
    clips: list[list[int]] = []

    full_segs = n_frames // (seq_len * stride)
    for i in range(full_segs):
        for j in range(stride):
            begin = i * (seq_len * stride) + j
            end = (i + 1) * (seq_len * stride)
            clip = list(range(begin, end, stride))
            clips.append(clip)

    remainder = n_frames % (seq_len * stride)
    if remainder != 0:
        base_offset = full_segs * (seq_len * stride)
        new_stride = remainder // seq_len
        for i in range(new_stride):
            begin = base_offset + i
            end = base_offset + seq_len * new_stride
            clip = list(range(begin, end, new_stride))
            clips.append(clip)

        if n_frames % seq_len != 0:
            start = (n_frames // seq_len) * seq_len
            clip = list(range(start, n_frames))
            while len(clip) < seq_len:
                clip = clip + clip
            clips.append(clip[:seq_len])

    # Fallback per tracklet molto corte
    if not clips:
        if n_frames >= seq_len:
            clips.append(list(range(seq_len)))
        else:
            clip = list(range(n_frames))
            while len(clip) < seq_len:
                clip = clip + clip
            clips.append(clip[:seq_len])

    return clips


def aggregate_embeddings(
    embs_list: list[torch.Tensor],
    min_intra_similarity: float | None = None,
) -> torch.Tensor | None:
    """Media embedding della stessa tracklet e normalizza L2.

    Se min_intra_similarity è impostato, scarta l'aggregato quando la similarità
    media tra le clip è troppo bassa (tracklet probabilmente rumorosa).
    """
    if not embs_list:
        return None

    stacked = torch.stack(embs_list)

    if min_intra_similarity is not None and len(embs_list) > 1:
        normed = torch.nn.functional.normalize(stacked, p=2, dim=1)
        cos_sim = torch.mm(normed, normed.t())
        n = cos_sim.size(0)
        mask = ~torch.eye(n, dtype=torch.bool)
        mean_sim = cos_sim[mask].mean().item()
        if mean_sim < min_intra_similarity:
            return None

    super_emb = torch.mean(stacked, dim=0)
    super_emb = torch.nn.functional.normalize(super_emb, p=2, dim=0)
    return super_emb
