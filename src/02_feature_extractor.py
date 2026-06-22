"""Modulo 2 — Feature Extraction (CAL / C2DResNet50) → Embedding.

Pipeline di elaborazione per l'estrazione di feature vettoriali (embeddings)
da tracklet video. Carica il backbone C2DResNet50 pre-addestrato con
Clothes-based Adversarial Loss (CAL) e applica la rete su batch di frame.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import (
    DEFAULT_REID_WEIGHTS,
    DEFAULT_SAMPLING_STRIDE,
    MIN_INTRA_TRACK_SIMILARITY,
    MIN_TRACK_FRAMES,
    PROJECT_ROOT,
    SEQ_LEN,
)
from src.utils import (
    EVAL_TRANSFORM,
    aggregate_embeddings,
    load_reid_model,
    recombine_tracklet_clips,
    setup_logging,
)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    _RICH = True
except ImportError:
    _RICH = False

log = logging.getLogger(__name__)


@dataclass
class Mod2VideoResult:
    """Risultato dell'elaborazione di un singolo video nel Modulo 2."""

    video: str
    identities: int = 0
    clips: int = 0
    dropped_short_tracks: int = 0
    elapsed_sec: float = 0.0
    skipped: bool = False


def parse_args() -> argparse.Namespace:
    """Esegue il parsing degli argomenti da riga di comando.

    Returns:
        argparse.Namespace: Oggetto contenente i parametri di configurazione
        come batch_size, directory di I/O e il numero di worker per il DataLoader.
    """
    p = argparse.ArgumentParser(description="Modulo 2 – Feature Extraction Re-ID con CAL")
    p.add_argument(
        "--input-dir", default=str(PROJECT_ROOT / "data" / "processed" / "extracted_rois")
    )
    p.add_argument("--weights", default=str(DEFAULT_REID_WEIGHTS))
    p.add_argument("--force", action="store_true", help="Rielabora video già completati")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    return p.parse_args()


class VideoTrackletDataset(Dataset):
    """Dataset custom per il caricamento ottimizzato delle tracklet video.

    Scansiona le directory alla ricerca di frame JPEG organizzati per track_id,
    e raggruppa i frame in clip temporali di dimensione 'seq_len'.
    """

    def __init__(self, video_dir: Path, seq_len: int = 8, transform=None):
        self.transform = transform
        self.chunks = []
        self.dropped_short_tracks = 0
        self.corrupted_frames = 0

        tracks = sorted(
            [d for d in video_dir.iterdir() if d.is_dir() and d.name.startswith("Track_")]
        )

        for track_dir in tracks:
            track_id = int(track_dir.name.split("_")[1])
            image_files = sorted(track_dir.glob("*.jpg"))
            num_frames = len(image_files)

            if num_frames < MIN_TRACK_FRAMES:
                self.dropped_short_tracks += 1
                continue

            # Genera gli indici temporali per raggruppare i frame in clip
            chunk_index_lists = recombine_tracklet_clips(
                num_frames,
                seq_len,
                DEFAULT_SAMPLING_STRIDE,
            )

            for indices in chunk_index_lists:
                chunk_paths = [image_files[idx % num_frames] for idx in indices]
                self.chunks.append((track_id, chunk_paths))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        track_id, img_paths = self.chunks[idx]
        valid_frames = []

        for p in img_paths:
            try:
                img = Image.open(p).convert("RGB")
            except (OSError, Image.UnidentifiedImageError) as exc:
                log.warning("Frame corrotto: %s — %s", p, exc)
                self.corrupted_frames += 1
                continue
            if self.transform:
                img = self.transform(img)
            valid_frames.append(img)

        if not valid_frames:
            # Clip interamente corrotta — restituisci zeri come fallback estremo
            # (il modello produrrà embedding nullo, ma non crasherà)
            zero_frame = (
                self.transform(Image.new("RGB", (128, 256), color=(128, 128, 128)))
                if self.transform
                else torch.zeros(3, 256, 128)
            )
            valid_frames = [zero_frame] * len(img_paths)
        elif len(valid_frames) < len(img_paths):
            # Duplica l'ultimo frame valido per arrivare a seq_len
            last = valid_frames[-1]
            while len(valid_frames) < len(img_paths):
                valid_frames.append(last)

        # [T, C, H, W] → [C, T, H, W] (layout atteso da C2DResNet)
        clip_tensor = torch.stack(valid_frames).permute(1, 0, 2, 3)
        return track_id, clip_tensor


def process_video_directory(
    model: torch.nn.Module,
    video_dir: Path,
    device: torch.device,
    force: bool,
    batch_size: int,
    workers: int,
) -> Mod2VideoResult:
    """Inferenza batch → mean pooling → normalizzazione L2 → salvataggio .pt."""
    video_name = video_dir.name
    out_pt = video_dir / "embeddings.pt"

    if out_pt.exists() and not force:
        log.info("Già elaborato, skip: %s", video_name)
        return Mod2VideoResult(video=video_name, skipped=True)

    t_start = time.time()

    dataset = VideoTrackletDataset(video_dir, seq_len=SEQ_LEN, transform=EVAL_TRANSFORM)
    dropped_short = dataset.dropped_short_tracks

    if len(dataset) == 0:
        return Mod2VideoResult(
            video=video_name,
            dropped_short_tracks=dropped_short,
        )

    log.info("Elaborazione %s (Totale %d mini-clip)...", video_name, len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    track_embs_accumulated = defaultdict(list)

    with torch.no_grad():
        for track_ids, clip_tensors in tqdm(dataloader, desc=f"  {video_name}", leave=False):
            clip_tensors = clip_tensors.to(device)
            feats = model(clip_tensors)

            for tid, feat in zip(track_ids, feats, strict=False):
                track_embs_accumulated[tid.item()].append(feat.cpu())

    embeddings_dict = {}
    for tid, embs_list in track_embs_accumulated.items():
        emb = aggregate_embeddings(
            embs_list,
            min_intra_similarity=MIN_INTRA_TRACK_SIMILARITY,
        )
        if emb is not None:
            embeddings_dict[tid] = emb
        else:
            log.debug("Track %d scartata per similarità intra-track bassa", tid)

    elapsed = time.time() - t_start

    if embeddings_dict:
        torch.save(embeddings_dict, out_pt)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return Mod2VideoResult(
        video=video_name,
        identities=len(embeddings_dict),
        clips=len(dataset),
        dropped_short_tracks=dropped_short,
        elapsed_sec=round(elapsed, 2),
    )


def _save_report(
    results: list[Mod2VideoResult],
    output_dir: Path,
    args: argparse.Namespace,
    elapsed: float,
) -> None:
    """Scrive ``pipeline_report_modulo2.json`` con statistiche aggregate."""
    report_path = output_dir / "pipeline_report_modulo2.json"
    report_path.write_text(
        json.dumps(
            {
                "elapsed_sec": round(elapsed, 2),
                "weights": args.weights,
                "batch_size": args.batch_size,
                "workers": args.workers,
                "seq_len": SEQ_LEN,
                "videos": [asdict(r) for r in results],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    log.info("Report Modulo 2 → %s", report_path)


def _print_rich_summary(results: list[Mod2VideoResult], elapsed: float) -> None:
    """Tabella Rich con riepilogo per video."""
    console = Console()
    table = Table(title="Riepilogo Modulo 2 (CAL)")
    table.add_column("Video", style="cyan")
    table.add_column("Identità", justify="right")
    table.add_column("Clip", justify="right")
    table.add_column("Scartate (corte)", justify="right")
    table.add_column("Tempo (s)", justify="right")
    table.add_column("Stato", justify="center")

    tot_tracks = 0
    for r in results:
        if r.skipped:
            table.add_row(r.video, "-", "-", "-", "-", "[yellow]skipped[/yellow]")
        else:
            tot_tracks += r.identities
            table.add_row(
                r.video,
                str(r.identities),
                str(r.clips),
                str(r.dropped_short_tracks),
                f"{r.elapsed_sec:.1f}",
                "[green]ok[/green]",
            )

    console.print(table)
    console.print(
        Panel(
            f"[bold]Identità totali estratte:[/bold] {tot_tracks}\n[bold]Tempo Totale:[/bold] {elapsed:.1f}s",
            title="Completato",
            border_style="green",
        )
    )


def main() -> None:
    setup_logging("run_mod2")
    args = parse_args()

    input_dir = Path(args.input_dir)
    weights_path = Path(args.weights)

    if not input_dir.exists():
        log.error("Cartella input non trovata: %s", input_dir)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(
        "Avvio Modulo 2 su device: %s | Batch Size: %d | Workers: %d",
        device,
        args.batch_size,
        args.workers,
    )

    try:
        model = load_reid_model(weights_path, device)
        log.info("Modello CAL caricato su %s.", device)
    except Exception:
        log.exception("Errore critico caricamento modello")
        sys.exit(1)

    video_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    t_start = time.time()
    results = []

    for v_dir in video_dirs:
        res = process_video_directory(
            model, v_dir, device, args.force, args.batch_size, args.workers
        )
        results.append(res)

    elapsed_tot = time.time() - t_start

    _save_report(results, input_dir, args, elapsed_tot)

    if _RICH and results:
        _print_rich_summary(results, elapsed_tot)
    else:
        log.info("Estrazione completata in %.1f secondi!", elapsed_tot)


if __name__ == "__main__":
    main()
