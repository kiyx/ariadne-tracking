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

    Criteri (allineati al paper MEVID: min 75px height, 25px width):
      1. Dimensione minima — crop troppo piccoli non hanno abbastanza dettaglio
      2. Aspect ratio massimo — w/h > MAX_ASPECT_RATIO indica un crop parziale
      3. Aspect ratio minimo — w/h < MIN_ASPECT_RATIO indica un artefatto
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
    """Rileva se una bbox tocca il bordo del frame (detection parziale).

    Una persona tagliata dal bordo del frame (top, bottom, left, right)
    produce crop inutili — solo piedi, solo testa, ecc.
    Restituisce True se la bbox tocca almeno un bordo entro il margine.

    NON filtra il bordo **superiore** né **laterale**: la testa tagliata
    in alto e persone al bordo laterale sono comuni e spesso ancora utili.
    Filtra solo il bordo **inferiore**: piedi tagliati dal basso indicano
    che la persona sta entrando/uscendo dal frame.
    """
    margin_y = int(frame_h * margin_ratio)
    # Solo bordo inferiore: se il bottom della bbox è al bordo del frame
    # E il top non è vicino al top del frame (→ non è una persona piena)
    at_bottom = y2 >= frame_h - margin_y
    at_top = y1 <= margin_y
    # Se tocca il bottom MA NON il top, è probabilmente solo la parte bassa
    return at_bottom and not at_top


def is_partial_body(
    box: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    kpt_conf: np.ndarray | None = None,
) -> bool:
    """Filtro ibrido pose-guided per detection parziali.

    Strategia a due livelli (stile MEVID pose-guided filtering):

    - **Detection grandi (h >= POSE_KPT_MIN_HEIGHT)**: i keypoint COCO sono
      affidabili → la detection è parziale se meno di POSE_MIN_UPPER_KEYPOINTS
      keypoint upper-body (indici 0-6: naso, occhi, orecchie, spalle) hanno
      confidenza >= POSE_KPT_CONF_THRESHOLD.

    - **Detection piccole (h < POSE_KPT_MIN_HEIGHT)** o senza keypoint:
      fallback a ``is_edge_bbox()`` (euristica geometrica sul bordo inferiore).

    Args:
        box:      Bounding box (x1, y1, x2, y2) in pixel.
        frame_w:  Larghezza del frame.
        frame_h:  Altezza del frame.
        kpt_conf: Array (17,) di confidenze per i 17 keypoint COCO,
                  oppure None se il modello non fornisce keypoint.

    Returns:
        True se la detection è parziale e va scartata.
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

    # Una bbox è soppressa se è strettamente più piccola nella coppia E ratio >= threshold.
    # Usiamo < invece di <= per evitare la soppressione reciproca quando le aree sono uguali.
    is_smaller = (
        areas[:, None] < areas[None, :]
    )  # (i, j): True se i è strettamente più piccola di j
    suppressed = np.any((ratio >= threshold) & is_smaller, axis=1)

    return cast("np.ndarray", ~suppressed)


def suppress_overlapping_boxes(
    boxes: np.ndarray,
    threshold: float = OVERLAP_IOU_THRESHOLD,
) -> np.ndarray:
    """Sopprime bbox sovrapposte con IoU > threshold (stile MEVID paper).

    Quando due detection si sovrappongono significativamente (IoU > 0.3),
    la più piccola viene scartata. Questo filtra le detection ambigue in
    scene affollate dove il tracker potrebbe confondere le identità.

    A differenza di ``suppress_contained_boxes`` (che richiede contenimento
    forte), questo filtro scatta anche con sovrapposizioni parziali.

    Args:
        boxes: Array (N, 4) in formato xyxy.
        threshold: Soglia IoU minima per sopprimere.

    Returns:
        Indici booleani (N,) — True = mantieni, False = sopprimi.
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

    # La bbox strettamente più piccola nella coppia viene soppressa.
    # Usiamo < invece di <= per evitare la soppressione reciproca quando le aree sono uguali.
    is_smaller = areas[:, None] < areas[None, :]
    suppressed = np.any((iou >= threshold) & is_smaller, axis=1)

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


# ══════════════════════════════════════════════════════════════
# ReID: utility condivise per sampling e aggregazione embedding
# ══════════════════════════════════════════════════════════════


@dataclass
class VideoMetadata:
    """Metadati estratti dal nome file video MEVID."""

    camera_id: str
    date: str
    scene_key: str
    camera_node_id: str
    location: str


def parse_video_metadata(video_name: str) -> VideoMetadata | None:
    """Estrae metadati temporali e spaziali dal nome file video MEVID.

    Format atteso: ``YYYY-MM-DD.HH-MM-SS.HH-MM-SS.location.G###.rN``
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
    """Restituisce il timestamp assoluto (epoch) di inizio del video.

    Estrae data e ora dal nome file MEVID. Se il parsing fallisce,
    restituisce 0.0 (fallback sicuro per differenze temporali).
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
    """Suddivide una tracklet in clip dense (protocollo reference CCVID).

    Replica ``_recombination_for_testset`` di simple_ccreid: genera tutte
    le clip possibili con stride temporale, coprendo l'intera sequenza.
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
    """Aggrega una lista di embedding con mean-pooling e L2-normalizzazione.

    Opzionalmente filtra per consistenza intra-track (cosine similarity media
    off-diagonal >= ``min_intra_similarity``).  Se il filtro fallisce, restituisce ``None``.

    Args:
        embs_list: Lista di tensori 1-D di embedding RAW (non normalizzati).
        min_intra_similarity: Se fornito, scarta l'aggregato se la similarità
            intra-track è inferiore alla soglia.

    Returns:
        Tensor L2-normalizzato [feat_dim] oppure ``None`` se scartato.
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
