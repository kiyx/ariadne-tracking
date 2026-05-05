"""Modulo 1 — Detection (YOLO26) + Tracking (BoT-SORT) → ROI.

Elabora una cartella di video, eseguendo person detection con YOLO26
e tracking con BoT-SORT.  Per ogni persona tracciata vengono estratte
le ROI (Region Of Interest) ritagliate e salvate come JPEG.

Input:
    Cartella di video (mp4, avi, mov, mkv, webm).

Output:
    Per ogni video → ``Track_XXXX/frame_YYYYYY.jpg`` + ``metadata.json``.
    In più, un ``pipeline_report_modulo1.json`` globale con statistiche aggregate.

Le ROI estratte alimentano il Modulo 2 (Simple-CCReID)
per il calcolo degli embedding di re-identificazione.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from src.config import (
    DEFAULT_CONF,
    DEFAULT_FRAME_SKIP,
    DEFAULT_IMGSZ,
    DEFAULT_INPUT_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    DEFAULT_TRACKER,
    JPEG_QUALITY,
    MIN_TRACK_FRAMES,
    PERSON_CLASS_ID,
    ROI_RESIZE,
    SHARPNESS_THRESHOLD,
    VIDEO_EXTENSIONS,
)
from src.utils import (
    VideoInfo,
    get_engine_path,
    get_video_info,
    is_partial_body,
    is_valid_roi,
    pad_and_clip_box,
    roi_sharpness,
    setup_logging,
    suppress_contained_boxes,
    suppress_overlapping_boxes,
)

# ── Rich (opzionale, per logging colorato e tabella riepilogo) ─

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    _RICH = True
except ImportError:
    _RICH = False

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Dataclass — risultati tipizzati
# ══════════════════════════════════════════════════════════════


@dataclass
class QualityStats:
    """Metriche di qualità aggregate delle ROI estratte."""

    avg_sharpness: float = 0.0
    good_rois: int = 0
    bad_rois: int = 0


@dataclass
class DiscardStats:
    """Conteggio delle ROI scartate, suddiviso per motivo.

    - ``frame_skip``     — scartate dalla regola 1-ogni-N.
    - ``invalid_roi``    — troppo piccole o con aspect ratio anomalo.
    - ``low_sharpness``  — ROI troppo sfocate (sotto soglia nitidezza).
    - ``imwrite_failed`` — errore di scrittura su disco.
    """

    frame_skip: int = 0
    invalid_roi: int = 0
    low_sharpness: int = 0
    imwrite_failed: int = 0
    contained: int = 0
    iou_overlap: int = 0
    edge_partial: int = 0


@dataclass
class VideoResult:
    """Risultato dell'elaborazione di un singolo video."""

    video: str
    frames: int = 0
    rois: int = 0
    tracks: int = 0
    elapsed_sec: float = 0.0
    processing_fps: float = 0.0
    quality: QualityStats = field(default_factory=QualityStats)
    discarded: DiscardStats = field(default_factory=DiscardStats)
    skipped: bool = False
    error: str | None = None


@dataclass
class ROIRecord:
    """Singolo record di una ROI salvata (finisce in ``metadata.json``)."""

    track_id: int
    frame_idx: int
    timestamp: float
    bbox_original: tuple[int, int, int, int]
    bbox_padded: tuple[int, int, int, int]
    confidence: float
    sharpness: float
    file: str


# ══════════════════════════════════════════════════════════════
# Eccezione di configurazione
# ══════════════════════════════════════════════════════════════


class ConfigError(ValueError):
    """Eccezione per parametri CLI non validi"""


# ══════════════════════════════════════════════════════════════
# Seed
# ══════════════════════════════════════════════════════════════


def set_seed(seed: int) -> None:
    """Fissa il seed su Python, NumPy e PyTorch per riproducibilità."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    """Definisce i parametri CLI e restituisce il Namespace parsato."""

    p = argparse.ArgumentParser(
        description="Modulo 1 – Detection + Tracking → ROI extraction",
    )

    # ── Path e modello ────────────────────────────────────────
    p.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Cartella video sorgente")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Cartella output ROI")
    p.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path modello YOLO (.pt)")

    # ── Parametri pipeline ────────────────────────────────────
    p.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Soglia confidence (0–1)")
    p.add_argument(
        "--frame-skip",
        type=int,
        default=DEFAULT_FRAME_SKIP,
        help="Salva 1 ROI ogni N frame",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help="Risoluzione YOLO (multiplo di 32)",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed riproducibilità")
    p.add_argument("--max-videos", type=int, default=0, help="Limita a N video (0 = tutti)")

    # ── Flag booleani ─────────────────────────────────────────
    p.add_argument("--no-resize", action="store_true", help="Non ridimensionare le ROI a 128×256")
    p.add_argument("--tensorrt", action="store_true", help="Usa TensorRT FP16 (richiede CUDA)")
    p.add_argument("--show", action="store_true", help="Preview detection in tempo reale")
    p.add_argument("--force", action="store_true", help="Rielabora video già completati")

    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Valida i parametri CLI.

    Raises:
        ConfigError: Se uno o più parametri non sono coerenti.
    """
    errors: list[str] = []

    if not (0.0 < args.conf <= 1.0):
        errors.append(f"--conf deve essere in (0, 1], ricevuto: {args.conf}")
    if args.frame_skip < 1:
        errors.append(f"--frame-skip deve essere >= 1, ricevuto: {args.frame_skip}")
    if args.imgsz < 32 or args.imgsz % 32 != 0:
        errors.append(f"--imgsz deve essere multiplo di 32, ricevuto: {args.imgsz}")
    if not Path(args.input_dir).is_dir():
        errors.append(f"Cartella di input inesistente: {args.input_dir}")
    if not Path(args.model).is_file():
        errors.append(f"Modello non trovato: {args.model}")
    if args.tensorrt and not torch.cuda.is_available():
        errors.append("--tensorrt richiede CUDA, ma torch.cuda non è disponibile")

    if errors:
        raise ConfigError("\n".join(errors))


# ══════════════════════════════════════════════════════════════
# Caricamento modello
# ══════════════════════════════════════════════════════════════


def load_model(model_path: str, imgsz: int, *, use_tensorrt: bool = False) -> YOLO:
    """Carica YOLO nella modalità più veloce disponibile.

    Priorità: engine TensorRT esistente → export TRT → PyTorch .pt.

    Args:
        model_path:   Path al file .pt del modello.
        imgsz:        Risoluzione di input (per il nome dell'engine).
        use_tensorrt: Se True, cerca/crea un engine TRT FP16.
    """
    engine = get_engine_path(model_path, imgsz)

    # Rileva automaticamente il task dal nome del modello
    task = "pose" if "pose" in Path(model_path).stem else "detect"
    log.info("Task YOLO rilevato: %s", task)

    if not use_tensorrt:
        log.info("Caricamento modello PyTorch: %s", model_path)
        return YOLO(model_path, task=task)

    if engine.exists():
        log.info("Caricamento motore TensorRT: %s", engine)
        return YOLO(str(engine), task=task)

    # Export TensorRT FP16 (una tantum, ~2-5 min)
    log.info("Export TensorRT FP16 in corso (una tantum)...")
    base = YOLO(model_path, task=task)
    exported = Path(
        base.export(format="engine", half=True, imgsz=imgsz, simplify=True, device=0),
    )
    if exported != engine:
        shutil.move(str(exported), str(engine))
    log.info("Export completato → %s", engine)
    return YOLO(str(engine), task=task)


# ══════════════════════════════════════════════════════════════
# Helper: estrazione e salvataggio ROI
# ══════════════════════════════════════════════════════════════


def _extract_roi(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    *,
    resize: bool = True,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Estrae una ROI dal frame: padding → validazione → crop → resize.

    Returns:
        ``(roi_image, padded_bbox)`` oppure ``None`` se la ROI è troppo
        piccola o ha aspect ratio anomalo.
    """
    px1, py1, px2, py2 = pad_and_clip_box(*box, frame_w, frame_h)

    if not is_valid_roi(px2 - px1, py2 - py1):
        return None

    roi = frame[py1:py2, px1:px2]
    if resize:
        roi = cv2.resize(roi, ROI_RESIZE, interpolation=cv2.INTER_CUBIC)
    return roi, (px1, py1, px2, py2)


def _save_roi_jpeg(roi: np.ndarray, filepath: Path) -> bool:
    """Scrive la ROI come JPEG con qualità configurata in ``config.py``."""
    return bool(cv2.imwrite(str(filepath), roi, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]))


def _compute_quality(sharpness_values: list[float]) -> QualityStats:
    """Calcola le statistiche di qualità dalle sharpness delle ROI salvate.

    Dopo il filtraggio hard (ROI sotto SHARPNESS_THRESHOLD scartate),
    le ROI rimaste vengono classificate in 'good' (> 2× soglia) o 'mediocre'.
    """
    if not sharpness_values:
        return QualityStats()
    avg = round(sum(sharpness_values) / len(sharpness_values), 1)
    # Soglia qualitativa "buona ROI" = 2x la soglia minima di filtro
    good_threshold = SHARPNESS_THRESHOLD * 2
    good = sum(1 for s in sharpness_values if s > good_threshold)
    return QualityStats(avg_sharpness=avg, good_rois=good, bad_rois=len(sharpness_values) - good)


# ══════════════════════════════════════════════════════════════
# Metadata I/O
# ══════════════════════════════════════════════════════════════


def _write_video_metadata(
    path: Path,
    video_name: str,
    info: VideoInfo,
    result: VideoResult,
    records: list[ROIRecord],
    status: str,
) -> None:
    """Scrive ``metadata.json`` in modo atomico (write-then-rename).

    Usa un file temporaneo ``.json.tmp`` e poi ``replace()`` per evitare
    che un crash lasci un file JSON parzialmente scritto.
    """
    data = {
        "video": video_name,
        "status": status,
        "resolution": [info.frame_w, info.frame_h],
        "fps": info.fps,
        "total_frames": info.total_frames,
        "processed_frames": result.frames,
        "total_rois": result.rois,
        "unique_tracks": result.tracks,
        "elapsed_sec": result.elapsed_sec,
        "processing_fps": result.processing_fps,
        "quality": asdict(result.quality),
        "discarded": asdict(result.discarded),
        "records": [asdict(r) for r in records],
    }
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


# ══════════════════════════════════════════════════════════════
# Elaborazione singolo video
# ══════════════════════════════════════════════════════════════


def process_single_video(
    video_path: str,
    model: YOLO,
    output_base_dir: Path,
    args: argparse.Namespace,
) -> VideoResult:
    """Pipeline completa per un singolo video: detection → tracking → ROI.

    Fasi:
      1. **Resume check** — salta se ``metadata.json`` esiste già.
      2. **Tracking** — ``model.track()``.
      3. **Estrazione ROI** — per ogni detection: padding, validazione,
         crop, resize, salvataggio JPEG.
      4. **Statistiche** — sharpness, contatori, scrittura metadata.
    """
    video_name = Path(video_path).stem
    video_out_dir = output_base_dir / video_name
    meta_path = video_out_dir / "metadata.json"

    # ── 1. Resume: salta video già elaborati ───────────────────
    if meta_path.exists() and not args.force:
        log.info("Già elaborato, skip: %s", video_name)
        return VideoResult(video=video_name, skipped=True)

    info = get_video_info(video_path)
    if info is None:
        log.error("Impossibile aprire: %s", video_path)
        return VideoResult(video=video_name, error="cannot_open")

    video_out_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "Inizio: %s (%d×%d @ %.1f fps, ~%d frame)",
        video_name,
        info.frame_w,
        info.frame_h,
        info.fps,
        info.total_frames,
    )

    # ── 2. Tracking  ────
    tracking_results = model.track(
        source=video_path,
        tracker=DEFAULT_TRACKER,
        classes=[PERSON_CLASS_ID],
        conf=args.conf,
        persist=True,
        stream=True,
        verbose=False,
        imgsz=args.imgsz,
        half=args.tensorrt,
        show=args.show,
    )

    # ── 3. Loop frame-by-frame ─────────────────────────────────
    discard = DiscardStats()
    records: list[ROIRecord] = []
    sharpness_vals: list[float] = []
    frame_counter: dict[int, int] = defaultdict(int)  # apparizioni per track
    created_tracks: set[int] = set()  # track con directory creata
    frame_idx = 0
    saved_rois = 0
    do_resize = not args.no_resize
    t_start = time.time()

    pbar = tqdm(total=info.total_frames, desc=f"  {video_name}", unit="fr", leave=True)
    interrupted = False

    try:
        for frame_result in tracking_results:
            frame = frame_result.orig_img

            # Se nessuna detection ha un track ID assegnato, skip al frame dopo
            if frame_result.boxes.id is not None:
                boxes = frame_result.boxes.xyxy.cpu().numpy()
                tids = frame_result.boxes.id.int().cpu().numpy()
                confs = frame_result.boxes.conf.cpu().numpy()

                # Keypoint COCO (17, conf) — disponibili solo con modello pose
                has_kpts = (
                    hasattr(frame_result, "keypoints")
                    and frame_result.keypoints is not None
                    and frame_result.keypoints.conf is not None
                )
                kpts_conf = (
                    frame_result.keypoints.conf.cpu().numpy() if has_kpts else None
                )  # (N, 17) o None

                # Sopprime bbox più piccole contenute in bbox più grandi
                # (es. piedi rilevati dentro una detection a corpo intero)
                keep_mask = suppress_contained_boxes(boxes)
                n_suppressed = int((~keep_mask).sum())
                if n_suppressed > 0:
                    discard.contained += n_suppressed
                    boxes = boxes[keep_mask]
                    tids = tids[keep_mask]
                    confs = confs[keep_mask]
                    if kpts_conf is not None:
                        kpts_conf = kpts_conf[keep_mask]

                # Sopprime detection con IoU > 0.3 (stile MEVID paper):
                # in scene affollate le detection sovrapposte sono ambigue
                if len(boxes) > 1:
                    iou_mask = suppress_overlapping_boxes(boxes)
                    n_iou = int((~iou_mask).sum())
                    if n_iou > 0:
                        discard.iou_overlap += n_iou
                        boxes = boxes[iou_mask]
                        tids = tids[iou_mask]
                        confs = confs[iou_mask]
                        if kpts_conf is not None:
                            kpts_conf = kpts_conf[iou_mask]

                for det_i, (box, tid_np, conf) in enumerate(
                    zip(boxes, tids, confs, strict=True),
                ):
                    tid = int(tid_np)
                    orig_box: tuple[int, int, int, int] = tuple(map(int, box))  # type: ignore[assignment]

                    # Frame-skip: il primo frame di ogni track è sempre salvato,
                    # poi solo ogni N-esimo (riduce il numero di ROI per track)
                    frame_counter[tid] += 1
                    if (
                        args.frame_skip > 1
                        and frame_counter[tid] > 1
                        and frame_counter[tid] % args.frame_skip != 0
                    ):
                        discard.frame_skip += 1
                        continue

                    # Filtra detection parziali: keypoint-guided per bbox grandi,
                    # fallback euristico (bordo inferiore) per bbox piccole
                    det_kpt = kpts_conf[det_i] if kpts_conf is not None else None
                    if is_partial_body(orig_box, info.frame_w, info.frame_h, det_kpt):
                        discard.edge_partial += 1
                        continue

                    # Estrai ROI (padding → validazione → crop → resize)
                    extraction = _extract_roi(
                        frame,
                        orig_box,
                        info.frame_w,
                        info.frame_h,
                        resize=do_resize,
                    )
                    if extraction is None:
                        discard.invalid_roi += 1
                        continue
                    roi, padded_box = extraction

                    # Filtra ROI troppo sfocate prima del salvataggio
                    sharp = roi_sharpness(roi)
                    if sharp < SHARPNESS_THRESHOLD:
                        discard.low_sharpness += 1
                        continue

                    # Directory track (creata lazy, una sola volta per ID)
                    track_dir_name = f"Track_{tid:04d}"
                    if tid not in created_tracks:
                        (video_out_dir / track_dir_name).mkdir(parents=True, exist_ok=True)
                        created_tracks.add(tid)

                    # Salva JPEG
                    filename = f"frame_{frame_idx:06d}.jpg"
                    filepath = video_out_dir / track_dir_name / filename
                    if not _save_roi_jpeg(roi, filepath):
                        log.warning("Scrittura fallita: %s", filepath)
                        discard.imwrite_failed += 1
                        continue

                    saved_rois += 1
                    sharpness_vals.append(sharp)

                    records.append(
                        ROIRecord(
                            track_id=tid,
                            frame_idx=frame_idx,
                            timestamp=round(frame_idx / info.fps, 4),
                            bbox_original=orig_box,
                            bbox_padded=padded_box,
                            confidence=round(float(conf), 4),
                            sharpness=round(sharp, 1),
                            file=f"{track_dir_name}/{filename}",
                        )
                    )

            frame_idx += 1
            pbar.update(1)
            pbar.set_postfix(roi=saved_rois)

    except KeyboardInterrupt:
        interrupted = True
        log.warning(
            "Ctrl+C — salvataggio parziale (%d/%d frame, %d ROI)",
            frame_idx,
            info.total_frames,
            saved_rois,
        )
    finally:
        pbar.close()

    # ── 4. Filtro track corte (allinea Modulo 1 con Modulo 2) ──
    if records:
        track_counts: dict[int, int] = defaultdict(int)
        for r in records:
            track_counts[r.track_id] += 1

        short_tracks = {tid for tid, count in track_counts.items() if count < MIN_TRACK_FRAMES}
        if short_tracks:
            for tid in short_tracks:
                track_dir = video_out_dir / f"Track_{tid:04d}"
                if track_dir.exists():
                    shutil.rmtree(track_dir)
            records = [r for r in records if r.track_id not in short_tracks]
            saved_rois = len(records)
            sharpness_vals = [r.sharpness for r in records]
            log.info(
                "%s: rimosse %d track corte (< %d frame)",
                video_name,
                len(short_tracks),
                MIN_TRACK_FRAMES,
            )

    # ── 5. Statistiche e metadata ──────────────────────────────
    elapsed = time.time() - t_start
    status = "interrupted" if interrupted else "completed"
    quality = _compute_quality(sharpness_vals)

    result = VideoResult(
        video=video_name,
        frames=frame_idx,
        rois=saved_rois,
        tracks=len({r.track_id for r in records}),
        elapsed_sec=round(elapsed, 2),
        processing_fps=round(frame_idx / elapsed, 1) if elapsed > 0 else 0.0,
        quality=quality,
        discarded=discard,
        error="interrupted" if interrupted else None,
    )

    _write_video_metadata(meta_path, video_name, info, result, records, status)
    log.info(
        "%s %s in %.1fs → %d ROI, %d track, %.1f fps (sharp=%.0f)",
        video_name,
        status,
        elapsed,
        saved_rois,
        result.tracks,
        result.processing_fps,
        quality.avg_sharpness,
    )
    return result


# ══════════════════════════════════════════════════════════════
# Report globale
# ══════════════════════════════════════════════════════════════


def _save_report(
    results: list[VideoResult],
    output_dir: Path,
    args: argparse.Namespace,
    elapsed: float,
) -> None:
    """Scrive ``pipeline_report.json`` e stampa riepilogo Rich (se disponibile)."""
    report_path = output_dir / "pipeline_report_modulo1.json"
    report_path.write_text(
        json.dumps(
            {
                "elapsed_sec": round(elapsed, 2),
                "model": args.model,
                "backend": "TensorRT FP16" if args.tensorrt else "PyTorch FP32",
                "engine": (str(get_engine_path(args.model, args.imgsz)) if args.tensorrt else None),
                "imgsz": args.imgsz,
                "conf_threshold": args.conf,
                "frame_skip": args.frame_skip,
                "resize": None if args.no_resize else list(ROI_RESIZE),
                "seed": args.seed,
                "videos": [asdict(r) for r in results],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    log.info("Pipeline terminata in %.1fs — report: %s", elapsed, report_path)

    if _RICH and results:
        _print_rich_summary(results, elapsed, report_path)


def _print_rich_summary(
    results: list[VideoResult],
    elapsed: float,
    report_path: Path,
) -> None:
    """Stampa una tabella Rich colorata con il riepilogo dei video."""
    console = Console()
    table = Table(title="Riepilogo Pipeline Modulo 1")
    table.add_column("Video", style="cyan")
    table.add_column("ROI", justify="right")
    table.add_column("Track", justify="right")
    table.add_column("FPS", justify="right")
    table.add_column("Sharp", justify="right")
    table.add_column("Qualità", justify="center")
    table.add_column("Stato", justify="center")

    total_rois = total_tracks = 0
    for r in results:
        if r.skipped:
            table.add_row(r.video, "-", "-", "-", "-", "-", "[yellow]skipped[/yellow]")
            continue

        total_rois += r.rois
        total_tracks += r.tracks
        q = r.quality
        total_q = q.good_rois + q.bad_rois
        ratio = f"{q.good_rois}/{total_q}" if total_q else "-"
        q_style = "green" if q.good_rois >= q.bad_rois else "red"

        if r.error == "interrupted":
            stato = "[yellow]interrupted[/yellow]"
        elif r.error:
            stato = f"[red]{r.error}[/red]"
        else:
            stato = "[green]ok[/green]"

        table.add_row(
            r.video,
            str(r.rois),
            str(r.tracks),
            f"{r.processing_fps:.1f}",
            f"{q.avg_sharpness:.0f}",
            f"[{q_style}]{ratio}[/{q_style}]",
            stato,
        )

    console.print(table)
    console.print(
        Panel(
            f"[bold]Totale:[/bold] {total_rois} ROI, {total_tracks} track, {elapsed:.1f}s\n"
            f"[bold]Report:[/bold] {report_path}",
            title="Completato",
            border_style="green",
        ),
    )


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════


def main() -> None:
    """Orchestratore: logging → seed → validazione → modello → loop video → report."""
    log_file = setup_logging("run_mod1")
    args = parse_args()

    try:
        validate_args(args)
    except ConfigError as exc:
        log.error(str(exc))
        sys.exit(1)

    set_seed(args.seed)
    if log_file:
        log.info("Log file: %s", log_file)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scansione video
    input_dir = Path(args.input_dir)
    video_files = sorted(
        f.name for f in input_dir.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not video_files:
        log.error("Nessun video in %s (formati: %s)", input_dir, ", ".join(VIDEO_EXTENSIONS))
        sys.exit(1)

    if args.max_videos > 0:
        video_files = video_files[: args.max_videos]

    # Caricamento modello
    model = load_model(args.model, args.imgsz, use_tensorrt=args.tensorrt)

    n = len(video_files)
    log.info("Trovati %d video da elaborare.", n)
    t_start = time.time()
    results: list[VideoResult] = []

    try:
        for i, vf in enumerate(video_files, 1):
            log.info("═══ Video %d/%d ═══", i, n)
            try:
                r = process_single_video(str(input_dir / vf), model, output_dir, args)
            except Exception:
                log.exception("Errore fatale su %s — skip", vf)
                r = VideoResult(video=Path(vf).stem, error="exception")
            results.append(r)
            if r.error == "interrupted":
                break
    except KeyboardInterrupt:
        log.warning("Interruzione manuale — salvataggio report parziale.")

    _save_report(results, output_dir, args, time.time() - t_start)


if __name__ == "__main__":
    main()
