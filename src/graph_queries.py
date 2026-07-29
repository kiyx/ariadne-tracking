"""Query e analisi statistiche sul grafo di movimento Ariadne.

Funzioni pure per interrogare global_graph.json e produrre statistiche su
percorsi, transizioni e attività delle telecamere. Nessuna dipendenza da UI.

Esempi:
    gq = GraphQueries(Path("output/global_graph.json"))
    results = gq.query_by_path(["G505", "G507"])
    stats = gq.get_transition_statistics()
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class PathMatch:
    """Risultato di query_by_path."""

    identity_id: str
    camera_sequence: list[str]
    node_sequence: list[str]
    times: list[float]
    track_ids: list[int]
    dates: list[str]
    is_cross_day: bool
    is_cross_camera: bool


@dataclass(frozen=True)
class TransitionStat:
    """Statistica di una transizione camera→camera."""

    src_camera: str
    dst_camera: str
    count: int
    avg_time_gap: float
    min_time_gap: float
    max_time_gap: float
    cross_day_count: int


@dataclass(frozen=True)
class CameraActivity:
    """Attività aggregata per una camera."""

    camera_id: str
    total_identities: int
    total_tracklets: int
    unique_dates: list[str]
    avg_identities_per_day: float


class GraphQueries:
    """Interfaccia di interrogazione del grafo globale.

    Carica il JSON una volta e costruisce indici per query rapide.
    """

    def __init__(self, graph_path: Path) -> None:
        self.graph_path = graph_path
        raw = json.loads(graph_path.read_text(encoding="utf-8"))

        self.identities: dict[str, list[dict]] = raw.get("identities", {})
        self.edges: list[dict] = raw.get("edges", [])
        self.nodes: list[dict] = raw.get("nodes", [])
        self.threshold: float = raw.get("threshold", 0.0)
        self.max_time_gap_sec: float = raw.get("max_time_gap_sec", 0.0)

        # Indici per query veloci
        self._identity_paths: dict[
            str, tuple[list[str], list[str], list[float], list[int], list[str]]
        ]
        self._identity_paths = {}
        self._build_index()

    def _build_index(self) -> None:
        """Costruisce indici percorso → identità per query rapide."""
        for iid, members in self.identities.items():
            if not members:
                continue
            members = sorted(members, key=lambda m: m["time"])
            cams = [m["camera_id"] for m in members]
            nodes = [m["node_id"] for m in members]
            times = [m["time"] for m in members]
            tids = [m["track_id"] for m in members]
            dates = [m.get("date", "?") for m in members]
            self._identity_paths[iid] = (cams, nodes, times, tids, dates)

    # --- Query per percorso ---

    def query_by_path(
        self,
        path_pattern: list[str],
        allow_subsequence: bool = True,
        date_filter: str | None = None,
    ) -> list[PathMatch]:
        """Trova identità che hanno percorso una sequenza di camere.

        Args:
            path_pattern: Lista di camera_id, es. ["G505", "G507", "G639"].
            allow_subsequence: Se True, accetta anche sotto-sequenze.
            date_filter: Se specificato, filtra per data.
        """
        if not path_pattern:
            return []

        results: list[PathMatch] = []

        for iid, (cams, nodes, times, tids, dates) in self._identity_paths.items():
            # Filtra per data se richiesto
            if date_filter and date_filter not in dates:
                continue

            # Cerca il pattern nella sequenza di camere
            match_idx = self._find_subsequence(cams, path_pattern, allow_subsequence)
            if match_idx is None:
                continue

            # Estrai la sottosequenza corrispondente
            end_idx = match_idx + len(path_pattern)
            sub_cams = cams[match_idx:end_idx]
            sub_nodes = nodes[match_idx:end_idx]
            sub_times = times[match_idx:end_idx]
            sub_tids = tids[match_idx:end_idx]
            sub_dates = dates[match_idx:end_idx]

            is_cross_day = len(set(sub_dates)) > 1
            is_cross_camera = len(set(sub_cams)) > 1

            results.append(
                PathMatch(
                    identity_id=iid,
                    camera_sequence=sub_cams,
                    node_sequence=sub_nodes,
                    times=sub_times,
                    track_ids=sub_tids,
                    dates=sub_dates,
                    is_cross_day=is_cross_day,
                    is_cross_camera=is_cross_camera,
                )
            )

        return results

    @staticmethod
    def _find_subsequence(
        seq: list[str],
        pattern: list[str],
        allow_subsequence: bool,
    ) -> int | None:
        """Primo indice in cui appare `pattern` in `seq`."""
        if not pattern:
            return 0

        for i in range(len(seq) - len(pattern) + 1):
            if seq[i : i + len(pattern)] == pattern:
                return i

        # Se allow_subsequence=True, cerca anche pattern parziali
        if allow_subsequence and len(pattern) > 1:
            # Cerca la sottosequenza più lunga possibile
            for i in range(len(seq)):
                matched = 0
                for j in range(i, len(seq)):
                    if matched < len(pattern) and seq[j] == pattern[matched]:
                        matched += 1
                    if matched == len(pattern):
                        return i

        return None

    # --- Query per identità ---

    def query_by_identity(self, identity_id: str) -> list[dict]:
        """Restituisce tutti gli avvistamenti di un'identità."""
        members = self.identities.get(identity_id, [])
        if not members:
            return []
        return sorted(members, key=lambda m: m["time"])

    def get_identity_summary(self, identity_id: str) -> dict | None:
        """Riepilogo strutturato di un'identità."""
        members = self.query_by_identity(identity_id)
        if not members:
            return None

        cams = [m["camera_id"] for m in members]
        nodes = [m["node_id"] for m in members]
        dates = [m.get("date", "?") for m in members]
        times = [m["time"] for m in members]

        return {
            "identity_id": identity_id,
            "num_sightings": len(members),
            "num_unique_cameras": len(set(cams)),
            "num_unique_days": len(set(dates)),
            "cameras": list(dict.fromkeys(cams)),  # preserva ordine
            "nodes": list(dict.fromkeys(nodes)),
            "dates": sorted(set(dates)),
            "start_time": min(times),
            "end_time": max(times),
            "duration_sec": max(times) - min(times),
            "is_cross_camera": len(set(cams)) > 1,
            "is_cross_day": len(set(dates)) > 1,
        }

    # --- Statistiche transizioni ---

    def get_transition_statistics(self) -> list[TransitionStat]:
        """Statistiche per ogni transizione camera→camera."""
        trans_counts: dict[tuple[str, str], int] = defaultdict(int)
        trans_gaps: dict[tuple[str, str], list[float]] = defaultdict(list)
        trans_cross_day: dict[tuple[str, str], int] = defaultdict(int)

        for edge in self.edges:
            src_cam = edge["src_camera"]
            dst_cam = edge["dst_camera"]
            key = (src_cam, dst_cam)

            trans_counts[key] += 1
            trans_gaps[key].append(edge["time_gap_sec"])
            if edge["src_date"] != edge["dst_date"]:
                trans_cross_day[key] += 1

        results: list[TransitionStat] = []
        for (src, dst), count in sorted(trans_counts.items()):
            gaps = trans_gaps[(src, dst)]
            results.append(
                TransitionStat(
                    src_camera=src,
                    dst_camera=dst,
                    count=count,
                    avg_time_gap=round(sum(gaps) / len(gaps), 1),
                    min_time_gap=round(min(gaps), 1),
                    max_time_gap=round(max(gaps), 1),
                    cross_day_count=trans_cross_day.get((src, dst), 0),
                )
            )

        return results

    def get_transition_matrix(
        self,
    ) -> dict[str, dict[str, dict[str, int]] | list[str] | dict[str, dict[str, float | None]]]:
        """Matrice transizioni camera×camera con conteggi e gap medi."""
        all_cams: list[str] = sorted(
            {edge["src_camera"] for edge in self.edges}
            | {edge["dst_camera"] for edge in self.edges}
        )

        matrix: dict[str, dict[str, int]] = {cam: dict.fromkeys(all_cams, 0) for cam in all_cams}
        gap_matrix: dict[str, dict[str, list[float]]] = {
            cam: {other: [] for other in all_cams} for cam in all_cams
        }

        for edge in self.edges:
            src = edge["src_camera"]
            dst = edge["dst_camera"]
            matrix[src][dst] += 1
            gap_matrix[src][dst].append(edge["time_gap_sec"])

        return {
            "cameras": all_cams,
            "counts": matrix,
            "avg_gaps": {
                cam: {
                    other: round(sum(gaps) / len(gaps), 1) if gaps else None
                    for other, gaps in gap_matrix[cam].items()
                }
                for cam in all_cams
            },
        }

    # --- Statistiche percorsi ---

    def get_path_statistics(self, top_k: int = 10) -> list[dict]:
        """Percorsi (sequenze di camere) più comuni tra le identità.

        Esclude percorsi mono-camera.
        """
        path_counts: Counter = Counter()
        path_durations: dict[tuple[str, ...], list[float]] = defaultdict(list)

        for _iid, (cams, _nodes, times, _tids, _dates) in self._identity_paths.items():
            if len(cams) < 2:
                continue
            # Semplifica: rimuovi camere consecutive duplicate
            simplified: list[str] = []
            for c in cams:
                if not simplified or c != simplified[-1]:
                    simplified.append(c)

            if len(simplified) < 2:
                continue

            path_key = tuple(simplified)
            path_counts[path_key] += 1
            path_durations[path_key].append(max(times) - min(times))

        results: list[dict] = []
        for path, count in path_counts.most_common(top_k):
            durations = path_durations[path]
            results.append(
                {
                    "path": " → ".join(path),
                    "path_list": list(path),
                    "count": count,
                    "avg_duration_sec": round(sum(durations) / len(durations), 1),
                    "num_cameras": len(path),
                }
            )

        return results

    # --- Attività per telecamera ---

    def get_camera_activity(self) -> list[CameraActivity]:
        """Attività aggregata per ogni camera fisica."""
        cam_stats: dict[str, dict] = defaultdict(
            lambda: {"identities": set(), "tracklets": 0, "dates": set()}
        )

        for node in self.nodes:
            cam = node["camera_id"]
            date = node.get("date", "?")
            cam_stats[cam]["dates"].add(date)
            cam_stats[cam]["tracklets"] += len(node.get("tracklets", []))

        for iid, members in self.identities.items():
            seen_cams = {m["camera_id"] for m in members}
            for cam in seen_cams:
                cam_stats[cam]["identities"].add(iid)

        results: list[CameraActivity] = []
        for cam, stats in sorted(cam_stats.items()):
            dates = sorted(stats["dates"])
            n_id = len(stats["identities"])
            results.append(
                CameraActivity(
                    camera_id=cam,
                    total_identities=n_id,
                    total_tracklets=stats["tracklets"],
                    unique_dates=dates,
                    avg_identities_per_day=round(n_id / max(len(dates), 1), 1),
                )
            )

        return results

    # --- Cross-day / Cross-camera stats ---

    def get_cross_statistics(self) -> dict:
        """Statistiche aggregate su cross-camera e cross-day."""
        cross_cam_identities = []
        cross_day_identities = []
        both_identities = []

        for iid, members in self.identities.items():
            if len(members) < 2:
                continue
            cams = {m["camera_id"] for m in members}
            dates = {m.get("date", "?") for m in members}
            is_cross_cam = len(cams) > 1
            is_cross_day = len(dates) > 1

            if is_cross_cam:
                cross_cam_identities.append(iid)
            if is_cross_day:
                cross_day_identities.append(iid)
            if is_cross_cam and is_cross_day:
                both_identities.append(iid)

        return {
            "total_identities": len(self.identities),
            "multi_member_identities": sum(1 for m in self.identities.values() if len(m) > 1),
            "cross_camera_identities": len(cross_cam_identities),
            "cross_day_identities": len(cross_day_identities),
            "cross_camera_and_day": len(both_identities),
            "cross_camera_ratio": round(
                len(cross_cam_identities) / max(len(self.identities), 1), 3
            ),
            "cross_day_ratio": round(len(cross_day_identities) / max(len(self.identities), 1), 3),
        }

    # --- Global metrics ---

    def get_global_metrics(self) -> dict:
        """Tutte le metriche globali del grafo."""
        return {
            "num_nodes": len(self.nodes),
            "num_identities": len(self.identities),
            "num_edges": len(self.edges),
            "threshold": self.threshold,
            "max_time_gap_sec": self.max_time_gap_sec,
            **self.get_cross_statistics(),
        }
