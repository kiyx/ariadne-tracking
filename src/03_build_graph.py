"""Modulo 3 — Grafo di Movimento Globale (Scene-Agnostic) con Clustering Gerarchico 3-Fasi.

Costruisce un grafo diretto pesato globale che collega tracklet
appartenenti alla stessa identità stimata, con archi orientati dal
primo al secondo avvistamento temporale.

Il grafo è **scene-agnostic**: il clustering avviene su tutti i video
disponibili, indipendentemente dalla scena. Ogni video costituisce un
nodo-telecamera univoco (camera + data), risolvendo esplicitamente il
problema del trattamento di giorni diversi per la stessa telecamera.

Clustering Gerarchico 3-Fasi
----------------------------
1. **Intra-Camera** (soglia alta): Fonde tracklet della stessa telecamera
   nello stesso giorno. Ripara frammentazione del tracker locale (BoT-SORT).
   Cannot-link con overlap temporale previene fusioni di persone diverse.

2. **Cross-Camera Same-Day** (soglia media): Fonde micro-cluster di
   telecamere diverse nello stesso giorno. Sfrutta continuità temporale
   e similarità visiva tra camere vicine.

3. **Cross-Day** (soglia bassa): Fonde meso-cluster di giorni diversi.
   Accetta variazioni di illuminazione e outfit change del dataset MEVID.

Algoritmo
---------
1. Carica gli embedding (Modulo 2) e i metadati temporali (Modulo 1).
2. Costruisce nodi: ogni video → nodo ``{camera_id}_{date}``.
3. Esegue clustering gerarchico 3-fasi (intra-camera → cross-camera → cross-day).
4. Genera archi diretti tra i membri di ogni cluster, orientati in
   ordine cronologico, con vincoli spazio-temporali.
5. Salva ``global_graph.json`` con topologia, identità e archi.

Partizionamento dinamico (per scena, per gruppi di telecamere) è
responsabilità del Modulo 4 (dashboard), che filtra il grafo globale
a query time e applica path simplification.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from scipy.cluster.hierarchy import fcluster, linkage

from src.config import PROJECT_ROOT
from src.utils import (
    get_video_absolute_start,
    load_all_embeddings,
    parse_video_metadata,
    setup_logging,
)

try:
    from rich.console import Console
    from rich.table import Table

    _RICH = True
except ImportError:
    _RICH = False

log = logging.getLogger(__name__)

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "extracted_rois"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Modulo 3 — Grafo di Movimento Globale (Clustering Gerarchico 3-Fasi)",
    )
    p.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Cartella con embeddings.pt e metadata.json per video",
    )
    p.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Cartella di output per il grafo JSON",
    )
    p.add_argument(
        "--threshold-intra",
        type=float,
        default=0.45,
        help="Soglia Fase 1 intra-camera (default: 0.45)",
    )
    p.add_argument(
        "--threshold-cross-cam",
        type=float,
        default=0.35,
        help="Soglia Fase 2 cross-camera same-day (default: 0.35)",
    )
    p.add_argument(
        "--threshold-cross-day",
        type=float,
        default=0.28,
        help="Soglia Fase 3 cross-day (default: 0.28)",
    )
    p.add_argument(
        "--min-gap-same-location",
        type=float,
        default=15.0,
        help="Min gap (s) per archi tra camere della stessa location (default: 15)",
    )
    p.add_argument(
        "--min-gap-diff-location",
        type=float,
        default=180.0,
        help="Min gap (s) per archi tra camere in location diverse (default: 180 = 3min)",
    )
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
# Caricamento metadati temporali
# ══════════════════════════════════════════════════════════════


def load_tracklet_times(
    input_dir: Path,
    video_names: list[str],
) -> dict[str, dict[int, float]]:
    """Carica il tempo mediano assoluto di ogni tracklet da metadata.json.

    Returns:
        Dict ``{video_name: {track_id: absolute_median_time_seconds}}``.
    """
    result: dict[str, dict[int, float]] = {}

    for video_name in video_names:
        meta_path = input_dir / video_name / "metadata.json"
        if not meta_path.exists():
            log.warning("metadata.json mancante per %s, skip.", video_name)
            continue

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            log.warning("metadata.json corrotto o illeggibile per %s, skip.", video_name)
            continue

        video_start = get_video_absolute_start(video_name)

        track_timestamps: dict[int, list[float]] = defaultdict(list)
        for record in meta.get("records", []):
            tid = record["track_id"]
            track_timestamps[tid].append(record["timestamp"])

        times: dict[int, float] = {}
        for tid, ts_list in track_timestamps.items():
            median_relative = float(np.median(ts_list))
            times[tid] = video_start + median_relative

        result[video_name] = times

    return result


def load_tracklet_time_ranges(
    input_dir: Path,
    video_names: list[str],
) -> dict[str, dict[int, tuple[float, float]]]:
    """Carica (start, end) assoluti di ogni tracklet da metadata.json.

    Returns:
        Dict ``{video_name: {track_id: (start_sec, end_sec)}}``.
    """
    result: dict[str, dict[int, tuple[float, float]]] = {}

    for video_name in video_names:
        meta_path = input_dir / video_name / "metadata.json"
        if not meta_path.exists():
            continue

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        video_start = get_video_absolute_start(video_name)

        track_timestamps: dict[int, list[float]] = defaultdict(list)
        for record in meta.get("records", []):
            tid = record["track_id"]
            track_timestamps[tid].append(record["timestamp"])

        ranges: dict[int, tuple[float, float]] = {}
        for tid, ts_list in track_timestamps.items():
            start = video_start + float(min(ts_list))
            end = video_start + float(max(ts_list))
            ranges[tid] = (start, end)

        result[video_name] = ranges

    return result


# ══════════════════════════════════════════════════════════════
# Clustering helpers
# ══════════════════════════════════════════════════════════════


def _cluster_group(
    indices: list[int],
    dist_matrix: np.ndarray,
    threshold: float,
) -> list[list[int]]:
    """Clustering agglomerativo su un sottoinsieme di tracklet.

    Args:
        indices: Indici dei nodi da clusterizzare.
        dist_matrix: Matrice distanza completa N×N.
        threshold: Soglia similarità (0-1).

    Returns:
        Lista di cluster, ogni cluster è una lista di indici originali.
    """
    if len(indices) <= 1:
        return [[i] for i in indices]

    sub_dist = dist_matrix[np.ix_(indices, indices)]
    n = len(indices)
    condensed = sub_dist[np.triu_indices(n, k=1)]
    Z = linkage(condensed, method="average")  # noqa: N806
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")

    clusters: dict[int, list[int]] = defaultdict(list)
    for local_idx, label in enumerate(labels):
        clusters[int(label)].append(indices[local_idx])
    return list(clusters.values())


def _cluster_clusters(
    clusters_list: list[list[int]],
    dist_matrix: np.ndarray,
    threshold: float,
) -> list[list[int]]:
    """Clustering a livello di cluster usando distanza originale (complete linkage).

    Evita il "blurring" della media degli embedding: misura la distanza
    MASSIMA tra qualsiasi coppia di punti dei due cluster sulla matrice
    distanza originale (complete linkage = conservativo).

    Args:
        clusters_list: Lista di cluster (ogni cluster = lista di indici).
        dist_matrix: Matrice distanza originale N×N tra tracklet.
        threshold: Soglia similarità (0-1).

    Returns:
        Lista di cluster fusi.
    """
    if len(clusters_list) <= 1:
        return clusters_list

    # Distanza completa tra cluster = max distanza tra qualsiasi coppia
    n = len(clusters_list)
    cluster_dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dists = dist_matrix[np.ix_(clusters_list[i], clusters_list[j])]
            cluster_dist[i, j] = float(np.max(dists))
            cluster_dist[j, i] = cluster_dist[i, j]

    np.fill_diagonal(cluster_dist, 0.0)
    condensed = cluster_dist[np.triu_indices(n, k=1)]
    Z = linkage(condensed, method="complete")  # noqa: N806
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")

    merged: dict[int, list[int]] = defaultdict(list)
    for cidx, label in enumerate(labels):
        for member in clusters_list[cidx]:
            merged[int(label)].append(member)
    return list(merged.values())


# ══════════════════════════════════════════════════════════════
# Costruzione grafo globale
# ══════════════════════════════════════════════════════════════


def build_global_graph(
    video_names: list[str],
    all_embeddings: dict[str, dict[int, torch.Tensor]],
    all_times: dict[str, dict[int, float]],
    all_time_ranges: dict[str, dict[int, tuple[float, float]]],
    threshold_intra: float,
    threshold_cross_cam: float,
    threshold_cross_day: float,
    min_gap_same_location: float,
    min_gap_diff_location: float,
) -> dict:
    """Costruisce il grafo diretto globale con clustering gerarchico 3-fasi.

    Returns:
        Dizionario con nodi, identità stimate, archi e statistiche.
    """
    # ── 1. Costruisci lista nodi (tracklet) ──
    nodes: list[dict] = []  # {video, track_id, camera_id, date, scene_key, node_id, location, time}
    embeddings: list[torch.Tensor] = []

    for video in video_names:
        video_embs = all_embeddings.get(video, {})
        video_times = all_times.get(video, {})
        meta = parse_video_metadata(video)

        camera_id = meta.camera_id if meta else "unknown"
        date = meta.date if meta else "unknown"
        scene_key = meta.scene_key if meta else "_ungrouped"
        node_id = meta.camera_node_id if meta else video
        location = meta.location if meta else "unknown"

        for tid, emb in video_embs.items():
            t = video_times.get(tid)
            if t is None:
                continue
            nodes.append(
                {
                    "video": video,
                    "track_id": tid,
                    "camera_id": camera_id,
                    "date": date,
                    "scene_key": scene_key,
                    "node_id": node_id,
                    "location": location,
                    "time": t,
                }
            )
            embeddings.append(emb)

    n = len(nodes)
    if n < 2:
        log.info("Solo %d tracklet nel dataset globale, skip.", n)
        return _empty_graph_result(n)

    log.info("Grafo globale: %d tracklet da %d video.", n, len(video_names))

    # ── 2. Matrice di similarità coseno ──
    emb_matrix = torch.stack(embeddings)
    emb_matrix = F.normalize(emb_matrix, p=2, dim=1)
    sim_matrix = torch.mm(emb_matrix, emb_matrix.t()).numpy()

    # ── 3. Matrice distanza per clustering ──
    dist_matrix = 1.0 - sim_matrix
    np.fill_diagonal(dist_matrix, 0.0)

    # ── 3b. Vincoli cannot-link ──
    cannot_link_penalty = 10.0
    for i in range(n):
        for j in range(i + 1, n):
            if nodes[i]["video"] == nodes[j]["video"]:
                vid = nodes[i]["video"]
                tid_i = nodes[i]["track_id"]
                tid_j = nodes[j]["track_id"]
                ranges = all_time_ranges.get(vid, {})
                start_i, end_i = ranges.get(tid_i, (0.0, float("inf")))
                start_j, end_j = ranges.get(tid_j, (0.0, float("inf")))
                if start_i < end_j and start_j < end_i:
                    dist_matrix[i, j] = cannot_link_penalty
                    dist_matrix[j, i] = cannot_link_penalty
                continue

            if nodes[i]["date"] == nodes[j]["date"]:
                time_diff = abs(nodes[i]["time"] - nodes[j]["time"])
                loc_a = nodes[i].get("location", "unknown")
                loc_b = nodes[j].get("location", "unknown")
                min_gap = min_gap_same_location if loc_a == loc_b else min_gap_diff_location
                if time_diff < min_gap:
                    dist_matrix[i, j] = cannot_link_penalty
                    dist_matrix[j, i] = cannot_link_penalty

    # ════════════════════════════════════════════════════════════
    # CLUSTERING GERARCHICO 3-FASI
    # ════════════════════════════════════════════════════════════

    # Fase 1: Intra-Camera (stessa camera + stessa data)
    node_ids = sorted({n["node_id"] for n in nodes})
    intra_clusters: list[list[int]] = []
    for nid in node_ids:
        indices = [i for i, n in enumerate(nodes) if n["node_id"] == nid]
        group_clusters = _cluster_group(indices, dist_matrix, threshold_intra)
        intra_clusters.extend(group_clusters)
    log.info(
        "Fase 1 intra-camera: %d micro-cluster (soglia %.2f)",
        len(intra_clusters),
        threshold_intra,
    )

    # Fase 2: Cross-Camera Same-Day (stessa data, camere diverse)
    dates = sorted({n["date"] for n in nodes})
    day_clusters: list[list[int]] = []
    for date in dates:
        day_micro = [c for c in intra_clusters if any(nodes[i]["date"] == date for i in c)]
        merged = _cluster_clusters(day_micro, dist_matrix, threshold_cross_cam)
        day_clusters.extend(merged)
    log.info(
        "Fase 2 cross-cam same-day: %d meso-cluster (soglia %.2f)",
        len(day_clusters),
        threshold_cross_cam,
    )

    # Fase 3: Cross-Day (date diverse)
    final_clusters = _cluster_clusters(day_clusters, dist_matrix, threshold_cross_day)
    log.info(
        "Fase 3 cross-day: %d macro-cluster (soglia %.2f)",
        len(final_clusters),
        threshold_cross_day,
    )

    clusters: dict[int, list[int]] = {i + 1: c for i, c in enumerate(final_clusters)}

    # ── 6. Genera archi diretti (ordinati per tempo) ──
    edges: list[dict] = []
    identities: dict[str, list[dict]] = {}
    identity_counter = 0

    for _label, member_indices in sorted(clusters.items()):
        members_sorted = sorted(member_indices, key=lambda i: nodes[i]["time"])

        identity_counter += 1
        identity_id = f"ID_{identity_counter:03d}"

        identity_members = []
        for i in members_sorted:
            node = nodes[i]
            identity_members.append(
                {
                    "node_id": node["node_id"],
                    "video": node["video"],
                    "camera_id": node["camera_id"],
                    "date": node["date"],
                    "scene_key": node["scene_key"],
                    "track_id": node["track_id"],
                    "time": round(node["time"], 1),
                }
            )

        identities[identity_id] = identity_members

        # Archi con vincoli spazio-temporali
        for a, b in pairwise(members_sorted):
            node_a, node_b = nodes[a], nodes[b]
            time_gap = node_b["time"] - node_a["time"]

            loc_a = node_a.get("location", "unknown")
            loc_b = node_b.get("location", "unknown")
            min_gap = min_gap_same_location if loc_a == loc_b else min_gap_diff_location
            if time_gap < min_gap:
                continue

            edges.append(
                {
                    "src": f"{node_a['node_id']}/Track_{node_a['track_id']:04d}",
                    "dst": f"{node_b['node_id']}/Track_{node_b['track_id']:04d}",
                    "src_node_id": node_a["node_id"],
                    "dst_node_id": node_b["node_id"],
                    "src_camera": node_a["camera_id"],
                    "dst_camera": node_b["camera_id"],
                    "src_date": node_a["date"],
                    "dst_date": node_b["date"],
                    "similarity": round(float(sim_matrix[a, b]), 4),
                    "time_gap_sec": round(time_gap, 1),
                    "src_time": round(node_a["time"], 1),
                    "dst_time": round(node_b["time"], 1),
                }
            )

    # ── 7. Costruisci nodi del grafo (video-telecamera) ──
    graph_nodes: dict[str, dict] = {}
    for node in nodes:
        nid = node["node_id"]
        if nid not in graph_nodes:
            graph_nodes[nid] = {
                "id": nid,
                "video": node["video"],
                "camera_id": node["camera_id"],
                "date": node["date"],
                "scene_key": node["scene_key"],
                "tracklets": [],
            }
        graph_nodes[nid]["tracklets"].append(
            {
                "track_id": node["track_id"],
                "time": round(node["time"], 1),
            }
        )

    # Statistiche
    n_singleton = sum(1 for m in clusters.values() if len(m) == 1)
    n_multi = sum(1 for m in clusters.values() if len(m) > 1)
    n_cross = sum(
        1
        for indices in clusters.values()
        if len(indices) > 1 and len({nodes[i]["camera_id"] for i in indices}) > 1
    )
    n_cross_day = sum(
        1
        for indices in clusters.values()
        if len(indices) > 1 and len({nodes[i]["date"] for i in indices}) > 1
    )
    n_cross_day_edges = sum(1 for e in edges if e["src_date"] != e["dst_date"])

    return {
        "version": "3.0",
        "scene_agnostic": True,
        "clustering": "hierarchical_3phase",
        "num_nodes": len(graph_nodes),
        "num_tracklets": n,
        "num_videos": len(video_names),
        "num_identities": len(clusters),
        "num_singletons": n_singleton,
        "num_multi_member": n_multi,
        "num_cross_camera": n_cross,
        "num_cross_day": n_cross_day,
        "num_cross_day_edges": n_cross_day_edges,
        "num_edges": len(edges),
        "threshold_intra": threshold_intra,
        "threshold_cross_cam": threshold_cross_cam,
        "threshold_cross_day": threshold_cross_day,
        "min_gap_same_location": min_gap_same_location,
        "min_gap_diff_location": min_gap_diff_location,
        "nodes": list(graph_nodes.values()),
        "identities": identities,
        "edges": edges,
    }


def _empty_graph_result(n_tracklets: int) -> dict:
    return {
        "version": "3.0",
        "scene_agnostic": True,
        "clustering": "hierarchical_3phase",
        "num_nodes": 0,
        "num_tracklets": n_tracklets,
        "num_videos": 0,
        "num_identities": n_tracklets,
        "num_singletons": n_tracklets,
        "num_multi_member": 0,
        "num_cross_camera": 0,
        "num_cross_day": 0,
        "num_edges": 0,
        "threshold_intra": 0.0,
        "threshold_cross_cam": 0.0,
        "threshold_cross_day": 0.0,
        "min_gap_same_location": 0.0,
        "min_gap_diff_location": 0.0,
        "nodes": [],
        "identities": {},
        "edges": [],
    }


# ══════════════════════════════════════════════════════════════
# Output
# ══════════════════════════════════════════════════════════════


def _print_summary(graph: dict) -> None:
    """Stampa riepilogo del grafo globale costruito."""
    if _RICH:
        console = Console()
        table = Table(title="Grafo di Movimento Globale — Riepilogo")
        table.add_column("Metrica", style="cyan")
        table.add_column("Valore", justify="right")

        metrics = [
            ("Nodi (video-telecamera)", graph["num_nodes"]),
            ("Tracklet", graph["num_tracklets"]),
            ("Identità", graph["num_identities"]),
            ("Singleton", graph["num_singletons"]),
            ("Multi-trk", graph["num_multi_member"]),
            ("Cross-camera", graph["num_cross_camera"]),
            ("Cross-day", graph["num_cross_day"]),
            ("Archi", graph["num_edges"]),
        ]
        for label, value in metrics:
            table.add_row(label, str(value))
        console.print(table)
    else:
        log.info(
            "Grafo globale: %d nodi, %d tracklet → %d identità (%d singleton, %d multi-trk, %d cross-cam, %d cross-day), %d archi",
            graph["num_nodes"],
            graph["num_tracklets"],
            graph["num_identities"],
            graph["num_singletons"],
            graph["num_multi_member"],
            graph["num_cross_camera"],
            graph["num_cross_day"],
            graph["num_edges"],
        )


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════


def main() -> None:
    setup_logging("build_graph")
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        log.error("Cartella input non trovata: %s", input_dir)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ── 1. Carica embedding ──
    all_embeddings = load_all_embeddings(input_dir)
    if not all_embeddings:
        log.error("Nessun embedding trovato in %s", input_dir)
        sys.exit(1)

    # ── 2. Carica metadati temporali ──
    all_video_names = list(all_embeddings.keys())
    all_times = load_tracklet_times(input_dir, all_video_names)
    log.info(
        "Metadati temporali caricati per %d/%d video.",
        len(all_times),
        len(all_video_names),
    )

    # ── 2b. Carica range temporali per cannot-link con overlap ──
    all_time_ranges = load_tracklet_time_ranges(input_dir, all_video_names)

    # ── 3. Costruisci grafo globale ──
    graph = build_global_graph(
        video_names=all_video_names,
        all_embeddings=all_embeddings,
        all_times=all_times,
        all_time_ranges=all_time_ranges,
        threshold_intra=args.threshold_intra,
        threshold_cross_cam=args.threshold_cross_cam,
        threshold_cross_day=args.threshold_cross_day,
        min_gap_same_location=args.min_gap_same_location,
        min_gap_diff_location=args.min_gap_diff_location,
    )

    # Salva grafo globale
    out_path = output_dir / "global_graph.json"
    out_path.write_text(
        json.dumps(graph, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Grafo globale → %s", out_path)

    elapsed = time.time() - t_start
    _print_summary(graph)
    log.info("Completato in %.1f s.", elapsed)


if __name__ == "__main__":
    main()
