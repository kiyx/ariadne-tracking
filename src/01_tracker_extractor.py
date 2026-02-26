"""Modulo 1 — Detection (YOLO26) + Tracking (BoT-SORT) → ROI.

Elabora una cartella di video, eseguendo person detection con YOLO26
e tracking con BoT-SORT.  Per ogni persona tracciata vengono estratte
le ROI (Region Of Interest) ritagliate e salvate come JPEG.

Input:
    Cartella di video (mp4, avi, mov, mkv, webm).

Output:
    Per ogni video → ``Track_XXXX/frame_YYYYYY.jpg`` + ``metadata.json``.
    In più, un ``pipeline_report.json`` globale con statistiche aggregate.

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
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from config import (
    DEFAULT_CONF,
    DEFAULT_FRAME_SKIP,
    DEFAULT_IMGSZ,
    DEFAULT_INPUT_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    DEFAULT_TRACKER,
    JPEG_QUALITY,
    LOG_DIR,
    PERSON_CLASS_ID,
    ROI_RESIZE,
    VIDEO_EXTENSIONS,
)
from utils import (
    get_engine_path,
    get_video_info,
    is_valid_roi,
    pad_and_clip_box,
    roi_sharpness,
)

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table

    _RICH = True
except ImportError:
    _RICH = False

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Logging (console + file)
# ══════════════════════════════════════════════════════════════


def setup_logging() -> Path | None:
    """
    Configura logging su console e su file.

    Il file di log viene salvato in output/logs/run_YYYYMMDD_HHMMSS.log
    così che ogni esecuzione abbia il suo log persistente e non si
    perdano informazioni se il terminale viene chiuso.

    Se ``rich`` è installato, la console usa colori e formattazione avanzata.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Formato per il file di log (sempre plain text)
    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler: rich se disponibile, altrimenti plain
    console_handler: logging.Handler
    if _RICH:
        console_handler = RichHandler(
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(file_fmt)

    root.addHandler(console_handler)

    # File handler
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOG_DIR / f"run_{timestamp}.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(file_fmt)
        root.addHandler(fh)
        return log_file
    except OSError:
        # Se non riesce a creare la directory/file di log, continua solo con console
        return None


# ══════════════════════════════════════════════════════════════
# Seed
# ══════════════════════════════════════════════════════════════


def set_seed(seed: int) -> None:
    """Imposta seed per riproducibilità su Python, NumPy e PyTorch.

    Args:
        seed: Valore intero del seed da applicare.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    """Definisce e restituisce i parametri da riga di comando.

    Returns:
        Namespace con tutti i flag CLI (input_dir, output_dir, model, conf, ecc.).
    """
    p = argparse.ArgumentParser(description="Modulo 1 – Detection + Tracking → ROI extraction")
    p.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Cartella con i video sorgente",
    )
    p.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Cartella di output per le ROI",
    )
    p.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path al modello YOLO (.pt)",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help="Soglia minima di confidenza (0.0–1.0)",
    )
    p.add_argument(
        "--frame-skip",
        type=int,
        default=DEFAULT_FRAME_SKIP,
        help="Salva 1 ROI ogni N frame per track",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help="Risoluzione di input per YOLO (multiplo di 32)",
    )
    p.add_argument(
        "--no-resize",
        action="store_true",
        help="Non ridimensionare le ROI a 128×256",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed per riproducibilità (default: 67)",
    )

    # ── Ottimizzazione GPU ────────────────────────────────────
    p.add_argument(
        "--tensorrt",
        action="store_true",
        help="Esporta in TensorRT FP16 e usa il motore ottimizzato",
    )

    # ── Opzioni avanzate ──────────────────────────────────────
    p.add_argument(
        "--show",
        action="store_true",
        help="Mostra finestra di preview con le detection in tempo reale",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rielabora video già completati (ignora metadata.json esistente)",
    )

    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Valida i parametri CLI; termina il processo con ``sys.exit(1)`` se non sono coerenti.

    Args:
        args: Namespace restituito da :func:`parse_args`.

    Raises:
        SystemExit: Se uno o più parametri non superano la validazione.
    """
    errors: list[str] = []

    if not (0.0 < args.conf <= 1.0):
        errors.append(f"--conf deve essere in (0.0, 1.0], ricevuto: {args.conf:.4f}")
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
        for msg in errors:
            log.error(msg)
        sys.exit(1)


# ══════════════════════════════════════════════════════════════
# Caricamento modello
# ══════════════════════════════════════════════════════════════


def load_model(args: argparse.Namespace) -> YOLO:
    """
    Carica il modello nella modalità più veloce disponibile.

    Priorità: TensorRT .engine già compilato > export TRT > PyTorch (.pt).
    """
    engine_path = get_engine_path(args.model, args.imgsz)

    if args.tensorrt:
        if engine_path.exists():
            log.info("Caricamento motore TensorRT: %s", engine_path)
            return YOLO(str(engine_path), task="detect")

        log.info("Export TensorRT FP16 in corso (una tantum, ~2-5 min)...")
        log.info("  Sorgente : %s", args.model)
        log.info("  imgsz    : %d", args.imgsz)
        log.info("  Output   : %s", engine_path)

        base_model = YOLO(args.model, task="detect")
        exported_path = base_model.export(
            format="engine",
            half=True,
            imgsz=args.imgsz,
            simplify=True,
            device=0,
        )

        exported = Path(exported_path)
        if exported != engine_path:
            shutil.move(str(exported), str(engine_path))

        log.info("Export completato → %s", engine_path)
        return YOLO(str(engine_path), task="detect")

    log.info("Caricamento modello PyTorch: %s", args.model)
    return YOLO(args.model, task="detect")


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════


def _empty_result(
    video_name: str,
    *,
    skipped: bool = False,
    error: str | None = None,
) -> dict[str, Any]:
    """Crea un risultato vuoto con struttura coerente per skip/errori."""
    return {
        "video": video_name,
        "frames": 0,
        "rois": 0,
        "tracks": 0,
        "elapsed_sec": 0,
        "processing_fps": 0,
        "quality": {"avg_sharpness": 0, "good_rois": 0, "bad_rois": 0},
        "discarded": {
            "low_conf": 0,
            "frame_skip": 0,
            "invalid_roi": 0,
            "imwrite_failed": 0,
        },
        "skipped": skipped,
        "error": error,
    }


# ══════════════════════════════════════════════════════════════
# Elaborazione singolo video
# ══════════════════════════════════════════════════════════════


def process_single_video(
    video_path: str,
    model: YOLO,
    output_base_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """
    Pipeline completa per un singolo video: detection → tracking → ROI.

    Output su disco:
      <output_base_dir>/<video_name>/Track_XXXX/frame_YYYYYY.jpg
      <output_base_dir>/<video_name>/metadata.json
    """
    video_name = Path(video_path).stem
    video_out_dir = output_base_dir / video_name

    # ── Resume: salta video già elaborati ──────────────────────
    meta_path = video_out_dir / "metadata.json"
    if meta_path.exists() and not args.force:
        log.info("Già elaborato (metadata.json presente), skip: %s", video_name)
        return _empty_result(video_name, skipped=True)

    video_out_dir.mkdir(parents=True, exist_ok=True)

    # ── Metadati video ─────────────────────────────────────────
    info = get_video_info(video_path)
    if info is None:
        log.error("Impossibile aprire %s", video_path)
        return _empty_result(video_name, error="cannot_open")

    frame_w: int = int(info["frame_w"])
    frame_h: int = int(info["frame_h"])
    fps: float = float(info["fps"])
    total_frames: int = int(info["total_frames"])

    log.info(
        "Inizio: %s  (%d×%d @ %.1f fps, ~%d frame)",
        video_name,
        frame_w,
        frame_h,
        fps,
        total_frames,
    )

    # ── Tracking ───────────────────────────────────────────────
    use_half = args.tensorrt

    results = model.track(
        source=video_path,
        tracker=DEFAULT_TRACKER,
        classes=[PERSON_CLASS_ID],
        conf=args.conf,
        persist=True,
        stream=True,
        verbose=False,
        imgsz=args.imgsz,
        half=use_half,
        show=args.show,
    )

    # ── Contatori e stato ──────────────────────────────────────
    frame_idx = 0
    saved_rois = 0
    track_frame_counter: dict[int, int] = defaultdict(int)
    created_dirs: set[str] = set()
    metadata_records: list[dict[str, Any]] = []

    skipped_low_conf = 0
    skipped_frame_skip = 0
    skipped_invalid_roi = 0
    failed_imwrite = 0
    sharpness_values: list[float] = []

    t_video_start = time.time()

    # Progress bar: rich se disponibile, altrimenti tqdm
    if _RICH:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("|"),
            TimeElapsedColumn(),
            TextColumn("|"),
            TimeRemainingColumn(),
            TextColumn("| ROI: {task.fields[roi_count]}"),
        )
        progress.start()
        task_id = progress.add_task(video_name, total=total_frames, roi_count=0)
    else:
        progress = None
        pbar = tqdm(total=total_frames, desc=f"  {video_name}", unit="fr", leave=True)

    # ── Loop frame-by-frame ────────────────────────────────────
    interrupted = False
    try:
        for frame_result in results:
            frame = frame_result.orig_img

            if frame_result.boxes.id is not None:
                boxes = frame_result.boxes.xyxy.cpu().numpy()
                track_ids = frame_result.boxes.id.int().cpu().numpy()
                confs = frame_result.boxes.conf.cpu().numpy()

                for box, track_id, conf in zip(boxes, track_ids, confs, strict=True):
                    x1, y1, x2, y2 = map(int, box)
                    tid = int(track_id)

                    # Filtro confidenza
                    if conf < args.conf:
                        skipped_low_conf += 1
                        continue

                    # Frame-skip per-track (il primo frame è sempre salvato)
                    track_frame_counter[tid] += 1
                    if (
                        args.frame_skip > 1
                        and track_frame_counter[tid] > 1
                        and track_frame_counter[tid] % args.frame_skip != 0
                    ):
                        skipped_frame_skip += 1
                        continue

                    # Padding + validazione
                    px1, py1, px2, py2 = pad_and_clip_box(
                        x1,
                        y1,
                        x2,
                        y2,
                        frame_w,
                        frame_h,
                    )
                    if not is_valid_roi(px2 - px1, py2 - py1):
                        skipped_invalid_roi += 1
                        continue

                    # Crop + resize
                    roi = frame[py1:py2, px1:px2]
                    if not args.no_resize:
                        roi = cv2.resize(roi, ROI_RESIZE, interpolation=cv2.INTER_LINEAR)

                    # Salvataggio JPEG
                    track_dir_name = f"Track_{tid:04d}"
                    track_dir = video_out_dir / track_dir_name
                    track_key = str(track_dir)
                    if track_key not in created_dirs:
                        track_dir.mkdir(parents=True, exist_ok=True)
                        created_dirs.add(track_key)

                    filename = f"frame_{frame_idx:06d}.jpg"
                    filepath = track_dir / filename
                    if not cv2.imwrite(
                        str(filepath),
                        roi,
                        [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)],
                    ):
                        log.warning("Scrittura fallita: %s", filepath)
                        failed_imwrite += 1
                        continue

                    saved_rois += 1

                    # Calcola nitidezza ROI
                    sharp = roi_sharpness(roi)
                    sharpness_values.append(sharp)

                    metadata_records.append(
                        {
                            "track_id": tid,
                            "frame_idx": frame_idx,
                            "timestamp": round(frame_idx / fps, 4),
                            "bbox_original": [x1, y1, x2, y2],
                            "bbox_padded": [px1, py1, px2, py2],
                            "confidence": round(float(conf), 4),
                            "sharpness": round(sharp, 1),
                            "file": f"{track_dir_name}/{filename}",
                        }
                    )

            frame_idx += 1
            if progress is not None:
                progress.update(task_id, advance=1, roi_count=saved_rois)
            else:
                pbar.update(1)
                pbar.set_postfix(roi=saved_rois)

    except KeyboardInterrupt:
        interrupted = True
        log.warning(
            "Ctrl+C su %s — salvataggio dati parziali (%d/%d frame, %d ROI).",
            video_name,
            frame_idx,
            total_frames,
            saved_rois,
        )
    finally:
        # Chiudi progress bar in ogni caso
        if progress is not None:
            progress.stop()
        else:
            pbar.close()

    elapsed = time.time() - t_video_start
    processing_fps = frame_idx / elapsed if elapsed > 0 else 0.0

    # ── Metadati strutturati ──
    unique_tracks = len({r["track_id"] for r in metadata_records})

    # Metriche di qualità
    avg_sharpness = (
        round(sum(sharpness_values) / len(sharpness_values), 1) if sharpness_values else 0
    )
    quality_good = sum(1 for s in sharpness_values if s > 100)
    quality_bad = len(sharpness_values) - quality_good

    status_label = "interrupted" if interrupted else "completed"

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "video": video_name,
                "status": status_label,
                "resolution": [frame_w, frame_h],
                "fps": fps,
                "total_frames": total_frames,
                "processed_frames": frame_idx,
                "total_rois": saved_rois,
                "unique_tracks": unique_tracks,
                "elapsed_sec": round(elapsed, 2),
                "processing_fps": round(processing_fps, 1),
                "quality": {
                    "avg_sharpness": avg_sharpness,
                    "good_rois": quality_good,
                    "bad_rois": quality_bad,
                },
                "discarded": {
                    "low_conf": skipped_low_conf,
                    "frame_skip": skipped_frame_skip,
                    "invalid_roi": skipped_invalid_roi,
                    "imwrite_failed": failed_imwrite,
                },
                "records": metadata_records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    log.info(
        "%s %s in %.1fs → %d ROI, %d track, %.1f fps  (sharp=%.0f)",
        video_name,
        status_label,
        elapsed,
        saved_rois,
        unique_tracks,
        processing_fps,
        avg_sharpness,
    )

    return {
        "video": video_name,
        "frames": frame_idx,
        "rois": saved_rois,
        "tracks": unique_tracks,
        "elapsed_sec": round(elapsed, 2),
        "processing_fps": round(processing_fps, 1),
        "quality": {
            "avg_sharpness": avg_sharpness,
            "good_rois": quality_good,
            "bad_rois": quality_bad,
        },
        "discarded": {
            "low_conf": skipped_low_conf,
            "frame_skip": skipped_frame_skip,
            "invalid_roi": skipped_invalid_roi,
            "imwrite_failed": failed_imwrite,
        },
        "skipped": False,
        "error": "interrupted" if interrupted else None,
    }


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════


def main() -> None:
    """Orchestratore: logging → seed → validazione → modello → loop video → report."""
    log_file = setup_logging()
    args = parse_args()
    validate_args(args)
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
        log.error(
            "Nessun video trovato in %s  (formati: %s)",
            args.input_dir,
            ", ".join(VIDEO_EXTENSIONS),
        )
        sys.exit(1)

    # Caricamento modello
    model = load_model(args)

    n_videos = len(video_files)
    log.info("Trovati %d video da elaborare.", n_videos)
    t_start = time.time()
    report: list[dict[str, Any]] = []

    try:
        for vid_idx, video_file in enumerate(video_files, start=1):
            log.info("═══ Video %d/%d ═══", vid_idx, n_videos)
            video_path = str(input_dir / video_file)
            try:
                stats = process_single_video(video_path, model, output_dir, args)
            except Exception:
                log.exception("Errore fatale su %s — skip", video_file)
                stats = _empty_result(Path(video_file).stem, error="exception")
            report.append(stats)
            if stats.get("error") == "interrupted":
                break
    except KeyboardInterrupt:
        log.warning("Interruzione manuale — salvataggio report parziale.")

    elapsed = time.time() - t_start

    # ── Report globale ────────────────────────────────────────
    _save_report(report, output_dir, args, elapsed)


def _save_report(
    report: list[dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
    elapsed: float,
) -> None:
    """Scrive pipeline_report.json e stampa il riepilogo rich."""
    engine_path = get_engine_path(args.model, args.imgsz)
    report_path = output_dir / "pipeline_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "elapsed_sec": round(elapsed, 2),
                "model": args.model,
                "backend": "TensorRT FP16" if args.tensorrt else "PyTorch FP32",
                "engine": str(engine_path) if args.tensorrt else None,
                "imgsz": args.imgsz,
                "conf_threshold": args.conf,
                "frame_skip": args.frame_skip,
                "resize": None if args.no_resize else list(ROI_RESIZE),
                "seed": args.seed,
                "videos": report,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    log.info("Pipeline Modulo 1 terminata in %.1fs — report: %s", elapsed, report_path)

    # ── Riepilogo con rich (se disponibile) ────────────────────
    if not (_RICH and report):
        return

    console = Console()
    table = Table(title="Riepilogo Pipeline Modulo 1")
    table.add_column("Video", style="cyan")
    table.add_column("ROI", justify="right")
    table.add_column("Track", justify="right")
    table.add_column("FPS", justify="right")
    table.add_column("Sharp", justify="right")
    table.add_column("Qualità", justify="center")
    table.add_column("Stato", justify="center")

    total_rois = 0
    total_tracks = 0
    for r in report:
        if r.get("skipped"):
            table.add_row(str(r["video"]), "-", "-", "-", "-", "-", "[yellow]skipped[/yellow]")
            continue

        rois: int = r.get("rois", 0)
        trks: int = r.get("tracks", 0)
        fps_val: float = r.get("processing_fps", 0)
        q: dict[str, Any] = r.get("quality", {})
        sharp: float = q.get("avg_sharpness", 0)
        good: int = q.get("good_rois", 0)
        bad: int = q.get("bad_rois", 0)
        ratio = f"{good}/{good + bad}" if (good + bad) > 0 else "-"
        q_style = "green" if good >= bad else "red"
        error = r.get("error")

        total_rois += rois
        total_tracks += trks

        if error == "interrupted":
            stato = "[yellow]interrupted[/yellow]"
        elif error:
            stato = f"[red]{error}[/red]"
        else:
            stato = "[green]ok[/green]"

        table.add_row(
            str(r["video"]),
            str(rois),
            str(trks),
            f"{fps_val:.1f}",
            f"{sharp:.0f}",
            f"[{q_style}]{ratio}[/{q_style}]",
            stato,
        )

    console.print(table)
    console.print(
        Panel(
            f"[bold]Totale:[/bold] {total_rois} ROI, {total_tracks} track, "
            f"{elapsed:.1f}s\n"
            f"[bold]Report:[/bold] {report_path}",
            title="Completato",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
