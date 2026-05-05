"""Validazione ReID su MEVID bbox_test (protocollo ufficiale).

Carica i crop ground-truth dal test split di MEVID, estrae le feature
con il backbone CAL (C2DResNet50), separa query e gallery secondo
``query_IDX.txt`` e calcola le metriche standard Re-ID:
CMC Rank-1/5/10/20 e mAP.

Protocollo di valutazione
-------------------------
- Ogni riga di ``track_test_info.txt`` definisce un tracklet con
  (start_idx, end_idx, person_id, outfit_id, camera_id).
- ``query_IDX.txt`` indica quali tracklet (per indice di riga) sono query.
- Le restanti tracklet formano la gallery.
- Per ogni query si escludono dalla gallery i campioni con stessa
  person_id E stessa camera_id (protocollo standard Re-ID).

Input:
    ``data/mevid-v1-bbox-test/bbox_test/{pid}/`` — crop GT per persona.
    ``data/mevid-v1-annotation-data/``           — file di annotazione.

Output:
    Metriche CMC + mAP stampate a console e salvate in JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import DEFAULT_REID_WEIGHTS, DEFAULT_SAMPLING_STRIDE, PROJECT_ROOT, SEQ_LEN
from src.utils import (
    EVAL_TRANSFORM,
    aggregate_embeddings,
    load_reid_model,
    recombine_tracklet_clips,
    setup_logging,
)

try:
    from rich.console import Console
    from rich.table import Table

    _RICH = True
except ImportError:
    _RICH = False

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# Strutture dati
# ══════════════════════════════════════════════════════════════


@dataclass
class Tracklet:
    """Metadati di una singola tracklet MEVID."""

    index: int  # riga in track_test_info.txt
    start_idx: int
    end_idx: int
    person_id: int
    outfit_id: int
    camera_id: int
    track_id: int  # T### nel nome file


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validazione Re-ID su MEVID bbox_test")
    p.add_argument(
        "--bbox-dir",
        default=str(PROJECT_ROOT / "data" / "mevid-v1-bbox-test" / "bbox_test"),
        help="Cartella con i crop GT per persona",
    )
    p.add_argument(
        "--annotation-dir",
        default=str(PROJECT_ROOT / "data" / "mevid-v1-annotation-data"),
        help="Cartella con track_test_info.txt, query_IDX.txt",
    )
    p.add_argument(
        "--weights",
        default=str(DEFAULT_REID_WEIGHTS),
    )
    p.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "output"),
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
# Caricamento annotazioni MEVID
# ══════════════════════════════════════════════════════════════


def load_tracklets(annotation_dir: Path) -> list[Tracklet]:
    """Legge track_test_info.txt e ricava il vero T### da test_name.txt."""
    info_path = annotation_dir / "track_test_info.txt"
    name_path = annotation_dir / "test_name.txt"

    # Carica tutti i nomi file per estrarre il T### reale
    with open(name_path, encoding="utf-8") as f:
        all_names = [line.strip() for line in f]

    rows: list[tuple[int, int, int, int, int]] = []
    with open(info_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            start, end, pid, oid, cid = (int(float(x)) for x in parts)
            rows.append((start, end, pid, oid, cid))

    _tid_re = re.compile(r"T(\d{3})")

    tracklets: list[Tracklet] = []
    for i, (start, end, pid, oid, cid) in enumerate(rows):
        # Estrai T### dal primo frame della tracklet in test_name.txt
        first_name = all_names[start]
        m = _tid_re.search(first_name)
        tid = int(m.group(1)) if m else 0
        tracklets.append(
            Tracklet(
                index=i,
                start_idx=start,
                end_idx=end,
                person_id=pid,
                outfit_id=oid,
                camera_id=cid,
                track_id=tid,
            )
        )
    return tracklets


def load_query_indices(annotation_dir: Path) -> set[int]:
    """Legge query_IDX.txt → set di indici (riga in track_test_info)."""
    idx_path = annotation_dir / "query_IDX.txt"
    indices: set[int] = set()
    with open(idx_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                indices.add(int(float(line)))
    return indices


# ══════════════════════════════════════════════════════════════
# Dataset per tracklet MEVID
# ══════════════════════════════════════════════════════════════


def tracklet_frame_paths(t: Tracklet, bbox_dir: Path) -> list[Path]:
    """Costruisce i path delle immagini per una tracklet.

    Il filename segue il formato MEVID:
    ``{pid:04d}O{oid:03d}C{cid:03d}T{tid:03d}F{fid:05d}.jpg``
    """
    n_frames = t.end_idx - t.start_idx + 1
    person_dir = bbox_dir / f"{t.person_id:04d}"
    paths = []
    for fid in range(n_frames):
        fname = (
            f"{t.person_id:04d}O{t.outfit_id:03d}C{t.camera_id:03d}T{t.track_id:03d}F{fid:05d}.jpg"
        )
        paths.append(person_dir / fname)
    return paths


class MevidTrackletDataset(Dataset):
    """Dataset che produce clip [C, T, H, W] per ogni tracklet MEVID.

    Usa sampling denso stride-based (protocollo reference CCVID) per coprire
    l'intera tracklet con clip sovrapposti.
    """

    def __init__(
        self,
        tracklets: list[Tracklet],
        bbox_dir: Path,
        seq_len: int = 8,
        stride: int = 4,
        transform=None,
    ):
        self.bbox_dir = bbox_dir
        self.transform = transform
        self.seq_len = seq_len
        self.clips: list[tuple[int, list[Path]]] = []  # (tracklet_index, frame_paths)

        for t in tracklets:
            all_paths = tracklet_frame_paths(t, bbox_dir)
            n_frames = len(all_paths)
            clip_indices_list = recombine_tracklet_clips(n_frames, seq_len, stride)

            for indices in clip_indices_list:
                frame_paths = [all_paths[i] for i in indices]
                self.clips.append((t.index, frame_paths))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        tracklet_idx, img_paths = self.clips[idx]
        frames = []
        for p in img_paths:
            img = Image.open(p).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        clip_tensor = torch.stack(frames).permute(1, 0, 2, 3)  # [C, T, H, W]
        return tracklet_idx, clip_tensor


# ══════════════════════════════════════════════════════════════
# Estrazione feature
# ══════════════════════════════════════════════════════════════


@torch.no_grad()
def extract_all_features(
    model: torch.nn.Module,
    tracklets: list[Tracklet],
    bbox_dir: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> dict[int, torch.Tensor]:
    """Estrae un embedding L2-normalizzato per ogni tracklet.

    Usa sampling denso stride-based e media delle feature RAW
    (normalizzazione L2 solo sul vettore finale), come nel reference.

    Returns:
        Dict {tracklet_index: embedding [2048]}.
    """
    dataset = MevidTrackletDataset(
        tracklets,
        bbox_dir,
        seq_len=SEQ_LEN,
        stride=DEFAULT_SAMPLING_STRIDE,
        transform=EVAL_TRANSFORM,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    log.info("Clip totali generati: %d (da %d tracklet)", len(dataset), len(tracklets))

    # Accumula feature RAW (non normalizzate) per tracklet, poi aggrega
    accum: dict[int, list[torch.Tensor]] = defaultdict(list)

    for batch_indices, batch_clips in tqdm(loader, desc="Estrazione feature", unit="batch"):
        batch_clips = batch_clips.to(device)  # [B, C, T, H, W]
        features = model(batch_clips)  # [B, feat_dim]
        # NON normalizzare qui — le feature raw vanno accumulate

        for tidx, feat in zip(batch_indices.tolist(), features.cpu(), strict=True):
            accum[tidx].append(feat)

    result: dict[int, torch.Tensor] = {}
    for tidx, feat_list in accum.items():
        emb = aggregate_embeddings(feat_list)
        if emb is not None:
            result[tidx] = emb

    return result


# ══════════════════════════════════════════════════════════════
# Metriche Re-ID (CMC + mAP)
# ══════════════════════════════════════════════════════════════


def compute_ap_cmc(
    index: np.ndarray, good_index: np.ndarray, junk_index: np.ndarray
) -> tuple[float, np.ndarray]:
    """Calcola AP e CMC per una singola query (standard Re-ID)."""
    ap = 0.0
    cmc = np.zeros(len(index))

    # Rimuovi indici junk (stessa persona + stessa camera)
    mask = np.isin(index, junk_index, invert=True)
    index = index[mask]

    ngood = len(good_index)
    if ngood == 0:
        return 0.0, cmc

    mask = np.isin(index, good_index)
    rows_good = np.argwhere(mask).flatten()

    if len(rows_good) == 0:
        return 0.0, cmc

    cmc[rows_good[0] :] = 1.0
    for i in range(ngood):
        if i < len(rows_good):
            d_recall = 1.0 / ngood
            precision = (i + 1) * 1.0 / (rows_good[i] + 1)
            ap += d_recall * precision

    return ap, cmc


def evaluate_reid(
    features: dict[int, torch.Tensor],
    tracklets: list[Tracklet],
    query_indices: set[int],
) -> dict:
    """Valutazione standard Re-ID: CMC Rank-1/5/10/20 e mAP.

    Per ogni query, la gallery esclude i campioni con stessa
    person_id E stessa camera_id (protocollo standard).
    """
    # Separa query e gallery
    q_indices = sorted([i for i in query_indices if i in features])
    g_indices = sorted(
        [i for i in range(len(tracklets)) if i not in query_indices and i in features]
    )

    if not q_indices or not g_indices:
        log.error("Query o gallery vuote! q=%d, g=%d", len(q_indices), len(g_indices))
        return {}

    log.info("Query: %d tracklet, Gallery: %d tracklet", len(q_indices), len(g_indices))

    # Costruisci tensori
    q_feats = torch.stack([features[i] for i in q_indices])
    g_feats = torch.stack([features[i] for i in g_indices])

    q_pids = np.array([tracklets[i].person_id for i in q_indices])
    g_pids = np.array([tracklets[i].person_id for i in g_indices])
    q_camids = np.array([tracklets[i].camera_id for i in q_indices])
    g_camids = np.array([tracklets[i].camera_id for i in g_indices])

    # Matrice distanze (1 - cosine_similarity)
    distmat = 1.0 - torch.mm(q_feats, g_feats.t()).numpy()

    num_q = distmat.shape[0]
    num_g = distmat.shape[1]
    index = np.argsort(distmat, axis=1)

    num_no_gt = 0
    CMC = np.zeros(num_g)  # noqa: N806
    AP = 0.0  # noqa: N806

    for i in range(num_q):
        query_index = np.argwhere(g_pids == q_pids[i]).flatten()
        camera_index = np.argwhere(g_camids == q_camids[i]).flatten()

        # Good: stessa persona, camera diversa
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if good_index.size == 0:
            num_no_gt += 1
            continue

        # Junk: stessa persona E stessa camera
        junk_index = np.intersect1d(query_index, camera_index)

        ap_tmp, cmc_tmp = compute_ap_cmc(index[i], good_index, junk_index)
        CMC += cmc_tmp  # noqa: N806
        AP += ap_tmp  # noqa: N806

    valid_queries = num_q - num_no_gt
    if valid_queries == 0:
        log.error("Nessuna query con GT valido nella gallery.")
        return {}

    if num_no_gt > 0:
        log.warning("%d query senza ground-truth nella gallery.", num_no_gt)

    CMC = CMC / valid_queries  # noqa: N806
    mAP = AP / valid_queries  # noqa: N806

    metrics = {
        "rank_1": round(float(CMC[0]) * 100, 2),
        "rank_5": round(float(CMC[4]) * 100, 2) if num_g > 4 else None,
        "rank_10": round(float(CMC[9]) * 100, 2) if num_g > 9 else None,
        "rank_20": round(float(CMC[19]) * 100, 2) if num_g > 19 else None,
        "mAP": round(float(mAP) * 100, 2),
        "num_query": num_q,
        "num_query_valid": valid_queries,
        "num_gallery": num_g,
        "num_no_gt": num_no_gt,
    }
    return metrics


# ══════════════════════════════════════════════════════════════
# Output
# ══════════════════════════════════════════════════════════════


def print_metrics(metrics: dict) -> None:
    if _RICH:
        console = Console()
        table = Table(title="MEVID Re-ID Evaluation")
        table.add_column("Metrica", style="cyan")
        table.add_column("Valore", style="green", justify="right")
        for k, v in metrics.items():
            if v is not None:
                table.add_row(k, str(v))
        console.print(table)
    else:
        print("\n=== MEVID Re-ID Evaluation ===")  # noqa: T201
        for k, v in metrics.items():
            if v is not None:
                print(f"  {k}: {v}")  # noqa: T201
        print()  # noqa: T201


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════


def main() -> None:
    setup_logging("eval_mevid")
    args = parse_args()

    bbox_dir = Path(args.bbox_dir)
    annotation_dir = Path(args.annotation_dir)
    weights_path = Path(args.weights)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not bbox_dir.exists():
        log.error("bbox_dir non trovata: %s", bbox_dir)
        sys.exit(1)
    if not annotation_dir.exists():
        log.error("annotation_dir non trovata: %s", annotation_dir)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── 1. Carica annotazioni ──
    log.info("Caricamento annotazioni MEVID...")
    tracklets = load_tracklets(annotation_dir)
    query_indices = load_query_indices(annotation_dir)
    log.info(
        "Tracklet totali: %d, Query: %d, Gallery: %d",
        len(tracklets),
        len(query_indices),
        len(tracklets) - len(query_indices),
    )

    # Verifica che alcuni file campione esistano
    sample_t = tracklets[0]
    sample_paths = tracklet_frame_paths(sample_t, bbox_dir)
    if not sample_paths[0].exists():
        log.error(
            "File campione non trovato: %s — Verifica il mapping T### dei tracklet.",
            sample_paths[0],
        )
        sys.exit(1)
    log.info("Verifica path OK: %s", sample_paths[0].name)

    # ── 2. Carica modello ──
    log.info("Caricamento modello CAL...")
    model = load_reid_model(weights_path, device)

    # ── 3. Estrai feature ──
    t0 = time.perf_counter()
    log.info("Estrazione feature per %d tracklet...", len(tracklets))
    features = extract_all_features(
        model=model,
        tracklets=tracklets,
        bbox_dir=bbox_dir,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    elapsed = time.perf_counter() - t0
    log.info("Feature estratte per %d tracklet in %.1f s.", len(features), elapsed)

    # ── 4. Valutazione ──
    log.info("Calcolo metriche Re-ID...")
    metrics = evaluate_reid(features, tracklets, query_indices)

    if not metrics:
        log.error("Valutazione fallita.")
        sys.exit(1)

    metrics["extraction_time_sec"] = round(elapsed, 1)
    metrics["sampling_stride"] = DEFAULT_SAMPLING_STRIDE

    print_metrics(metrics)

    # Salva JSON
    out_path = output_dir / "eval_mevid_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log.info("Risultati salvati in %s", out_path)


if __name__ == "__main__":
    main()
