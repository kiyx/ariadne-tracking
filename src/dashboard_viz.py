"""Visualizzazioni Plotly + pyvis per la dashboard Ariadne.

Funzioni pure (senza dipendenza da Streamlit) per renderizzare figure e grafi.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

if TYPE_CHECKING:
    from pathlib import Path

    from plotly.graph_objects import Figure

    from src.graph_queries import GraphQueries

# --- Palette ---

_BG_DARK = "#0b1220"
_BG_CARD = "#111b2e"
_BG_LIGHT = "#162033"
_TEXT_MAIN = "#f8fafc"
_TEXT_MUTED = "#94a3b8"
_ACCENT = "#14b8a6"
_ACCENT_LIGHT = "#5eead4"
_ACCENT_WARN = "#f59e0b"
_ACCENT_DANGER = "#ef4444"
_BORDER = "#1e293b"

_CAM_PALETTE = [
    "#14b8a6",
    "#60a5fa",
    "#f472b6",
    "#a78bfa",
    "#fbbf24",
    "#fb923c",
    "#22d3ee",
    "#f87171",
    "#a3e635",
    "#e879f9",
]

# --- Heatmap ---


def render_heatmap(gq: GraphQueries, use_avg_gap: bool = False) -> Figure:
    """Heatmap transizioni camera→camera."""
    matrix = gq.get_transition_matrix()
    cams: list[str] = matrix["cameras"]  # type: ignore
    counts: dict[str, dict[str, int]] = matrix["counts"]  # type: ignore
    avg_gaps: dict[str, dict[str, float | None]] = matrix["avg_gaps"]  # type: ignore

    if not cams:
        return go.Figure()

    n = len(cams)
    z_values: list[list[float]] = []
    hover_texts: list[list[str]] = []

    for src in cams:
        row_z: list[float] = []
        row_hover: list[str] = []
        for dst in cams:
            count = counts[src][dst]
            gap = avg_gaps[src][dst]
            val = gap if use_avg_gap and gap is not None else float(count)
            row_z.append(val)
            gap_str = f"{gap:.1f}s" if gap is not None else "—"
            row_hover.append(
                f"<b>{src}</b> → <b>{dst}</b><br>"
                f"Transizioni: <span style='color:#14b8a6'>{count}</span><br>"
                f"Gap medio: {gap_str}"
            )
        z_values.append(row_z)
        hover_texts.append(row_hover)

    title = "Gap Temporale Medio (s)" if use_avg_gap else "Numero Transizioni"

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=cams,
            y=cams,
            text=[[str(int(v)) for v in row] for row in z_values],
            texttemplate="%{text}",
            textfont={"size": 11, "color": _TEXT_MAIN},
            hoverongaps=False,
            hoverinfo="text",
            hovertext=hover_texts,
            colorscale="Teal",
            colorbar={
                "title": {"text": title, "font": {"color": _TEXT_MUTED, "size": 11}},
                "tickfont": {"color": _TEXT_MUTED},
                "bgcolor": _BG_CARD,
                "thickness": 18,
            },
        )
    )

    fig.update_layout(
        title={
            "text": f"<b>Heatmap</b>  <span style='color:#94a3b8;font-size:13px'>{title}</span>",
            "font": {"size": 16, "color": _TEXT_MAIN},
            "x": 0.5,
        },
        xaxis={"side": "top", "tickfont": {"color": _TEXT_MUTED, "size": 10}, "gridcolor": _BORDER},
        yaxis={"tickfont": {"color": _TEXT_MUTED, "size": 10}, "gridcolor": _BORDER},
        height=max(400, n * 50),
        width=max(500, n * 50),
        plot_bgcolor=_BG_DARK,
        paper_bgcolor=_BG_DARK,
        font={"color": _TEXT_MAIN, "family": "Segoe UI, system-ui, sans-serif"},
        margin={"t": 70, "b": 50, "l": 70, "r": 40},
    )
    return fig


def render_sankey(gq: GraphQueries, top_k: int | None = None) -> Figure:
    """Sankey dei flussi camera→camera (spessore = n. transizioni)."""
    matrix = gq.get_transition_matrix()
    cams: list[str] = matrix["cameras"]  # type: ignore
    counts: dict[str, dict[str, int]] = matrix["counts"]  # type: ignore
    avg_gaps: dict[str, dict[str, float | None]] = matrix["avg_gaps"]  # type: ignore
    if not cams:
        return go.Figure()

    pairs = [
        (src, dst, counts[src][dst], avg_gaps[src][dst])
        for src in cams
        for dst in cams
        if counts[src][dst] > 0
    ]
    pairs.sort(key=lambda p: p[2], reverse=True)
    if top_k is not None:
        pairs = pairs[:top_k]
    if not pairs:
        return go.Figure()

    idx = {c: i for i, c in enumerate(cams)}
    colors = [_CAM_PALETTE[i % len(_CAM_PALETTE)] for i in range(len(cams))]
    fig = go.Figure(
        data=go.Sankey(
            node={
                "label": cams,
                "color": colors,
                "line": {"color": _BORDER, "width": 1},
            },
            link={
                "source": [idx[s] for s, _, _, _ in pairs],
                "target": [idx[d] for _, d, _, _ in pairs],
                "value": [c for _, _, c, _ in pairs],
                "customdata": [g if g is not None else "—" for _, _, _, g in pairs],
                "color": ["rgba(20,184,166,0.25)"] * len(pairs),
                "hovertemplate": (
                    "%{source.label} → %{target.label}<br>"
                    "Transizioni: %{value}<br>Gap medio: %{customdata}s<extra></extra>"
                ),
            },
        )
    )
    fig.update_layout(
        title={
            "text": "<b>Flussi</b>  <span style='color:#94a3b8;font-size:13px'>transizioni camera→camera</span>",
            "font": {"size": 16, "color": _TEXT_MAIN},
            "x": 0.5,
        },
        height=520,
        plot_bgcolor=_BG_DARK,
        paper_bgcolor=_BG_DARK,
        font={"color": _TEXT_MAIN, "family": "Segoe UI, system-ui, sans-serif"},
        margin={"t": 70, "b": 30, "l": 30, "r": 30},
    )
    return fig


# --- Grafo interattivo pyvis ---


def render_camera_graph(
    gq: GraphQueries,
    highlight_path: list[str] | None = None,
    highlight_identity: str | None = None,
    min_edge_weight: int = 1,
    physics: bool = True,
    group_by_date: bool = False,
) -> str:
    """HTML del grafo interattivo PyVis."""
    from pyvis.network import Network

    if highlight_identity:
        summary = gq.get_identity_summary(highlight_identity)
        if summary:
            highlight_path = summary["cameras"]

    graph = nx.DiGraph()

    if group_by_date:
        # Vista per data: un nodo per camera/giorno
        all_cams = sorted({n["camera_id"] for n in gq.nodes})
        cam_colors = {c: _CAM_PALETTE[i % len(_CAM_PALETTE)] for i, c in enumerate(all_cams)}

        for node in gq.nodes:
            nid = node["id"]
            cam = node["camera_id"]
            date = node.get("date", "?")
            n_tracklets = len(node.get("tracklets", []))
            # Label abbreviata: es. G505\n(05-18)
            label = f"{cam}\n({date[5:] if len(date) >= 10 else date})"
            graph.add_node(
                nid,
                label=label,
                color=cam_colors.get(cam, "#888"),
                size=max(20, min(80, n_tracklets * 1.5)),
                title=(
                    f"<b>{nid}</b><br>Camera: {cam}<br>Date: {date}<br>Tracklets: {n_tracklets}"
                ),
            )

        # Archi diretti da edges usando src_node_id / dst_node_id
        edge_counts: dict[tuple[str, str], dict] = {}
        for edge in gq.edges:
            src = edge["src_node_id"]
            dst = edge["dst_node_id"]
            key = (src, dst)
            if key not in edge_counts:
                edge_counts[key] = {"count": 0, "gaps": [], "cross_day": 0}
            edge_counts[key]["count"] += 1
            edge_counts[key]["gaps"].append(edge["time_gap_sec"])
            if edge["src_date"] != edge["dst_date"]:
                edge_counts[key]["cross_day"] += 1

        for (src, dst), data in edge_counts.items():
            if data["count"] < min_edge_weight:
                continue
            avg_gap = sum(data["gaps"]) / len(data["gaps"])
            graph.add_edge(
                src,
                dst,
                weight=data["count"],
                label=str(data["count"]),
                title=(
                    f"<b>{data['count']}</b> transizioni<br>"
                    f"Gap: {avg_gap:.1f}s<br>"
                    f"Cross-day: {data['cross_day']}"
                ),
            )

        # Evidenzia i nodi la cui camera è nel percorso
        if highlight_path:
            for node in gq.nodes:
                nid = node["id"]
                cam = node["camera_id"]
                if cam in highlight_path and nid in graph.nodes:
                    graph.nodes[nid]["borderWidth"] = 4
                    graph.nodes[nid]["borderWidthSelected"] = 6
                    graph.nodes[nid]["color"] = {
                        "background": cam_colors.get(cam, "#888"),
                        "border": _ACCENT_WARN,
                    }
    else:
        # Vista collassata: un nodo per camera
        stats = gq.get_transition_statistics()
        activity = {a.camera_id: a for a in gq.get_camera_activity()}
        all_cams = sorted(activity)
        cam_colors = {c: _CAM_PALETTE[i % len(_CAM_PALETTE)] for i, c in enumerate(all_cams)}

        for cam in all_cams:
            act = activity[cam]
            graph.add_node(
                cam,
                label=f"{cam}",
                color=cam_colors.get(cam, "#888"),
                size=max(20, min(80, act.total_tracklets * 1.5)),
                title=(
                    f"<b>{cam}</b><br>"
                    f"Identità: {act.total_identities}<br>"
                    f"Tracklets: {act.total_tracklets}<br>"
                    f"Giorni: {', '.join(act.unique_dates)}"
                ),
            )

        for stat in stats:
            if stat.count < min_edge_weight:
                continue
            graph.add_edge(
                stat.src_camera,
                stat.dst_camera,
                weight=stat.count,
                label=str(stat.count),
                title=(
                    f"<b>{stat.count}</b> transizioni<br>"
                    f"Gap: {stat.avg_time_gap:.1f}s<br>"
                    f"Cross-day: {stat.cross_day_count}"
                ),
            )

        if highlight_path:
            for cam in highlight_path:
                if cam in graph.nodes:
                    graph.nodes[cam]["borderWidth"] = 4
                    graph.nodes[cam]["borderWidthSelected"] = 6
                    graph.nodes[cam]["color"] = {
                        "background": cam_colors.get(cam, "#888"),
                        "border": _ACCENT_WARN,
                    }

    net = Network(
        height="550px",
        width="100%",
        bgcolor=_BG_DARK,
        font_color=True,  # pyright: ignore[reportArgumentType]
    )
    net.from_nx(graph)

    for node in net.nodes:
        node["font"] = {"size": 14, "color": _TEXT_MAIN, "face": "Segoe UI, sans-serif"}
        node["shape"] = "dot"
        node["shadow"] = {
            "enabled": True,
            "color": "rgba(20,184,166,0.2)",
            "size": 10,
            "x": 3,
            "y": 3,
        }

    for edge in net.edges:
        w = edge.get("weight", 1)
        edge["width"] = max(1, min(8, w * 0.6))
        edge["color"] = {"color": "#475569", "highlight": _ACCENT, "hover": _ACCENT_LIGHT}
        edge["arrows"] = {"to": {"enabled": True, "scaleFactor": 0.8}}
        edge["smooth"] = {"type": "dynamic", "roundness": 0.3}

    # Evidenzia gli archi che fanno parte di un percorso richiesto
    if highlight_path:
        path_pairs = {
            (highlight_path[i], highlight_path[i + 1]) for i in range(len(highlight_path) - 1)
        }
        nid_to_cam = {n["id"]: n["camera_id"] for n in gq.nodes} if group_by_date else {}
        for edge in net.edges:
            src = edge["from"]
            dst = edge["to"]
            src_cam = nid_to_cam.get(src, src)
            dst_cam = nid_to_cam.get(dst, dst)
            if (src_cam, dst_cam) in path_pairs:
                edge["color"] = {
                    "color": _ACCENT_WARN,
                    "highlight": _ACCENT_WARN,
                    "hover": _ACCENT_LIGHT,
                }
                edge["width"] = max(edge.get("width", 1), 4)

    net.set_options(
        json.dumps(
            {
                "layout": {"randomSeed": 7, "improvedLayout": True},
                "physics": {
                    "enabled": physics,
                    "forceAtlas2Based": {
                        "gravitationalConstant": -80,
                        "centralGravity": 0.01,
                        "springLength": 180,
                        "springConstant": 0.25,
                        "damping": 0.4,
                    },
                    "maxVelocity": 50,
                    "solver": "forceAtlas2Based",
                    "timestep": 0.3,
                    "stabilization": {"enabled": True, "iterations": 1000, "updateInterval": 25},
                },
                "interaction": {"hover": True, "tooltipDelay": 150, "hideEdgesOnDrag": True},
            }
        )
    )
    return str(net.generate_html())


# --- Timeline ---


def render_timeline(
    gq: GraphQueries,
    highlight_identity: str | None = None,
    group_by_date: bool = False,
) -> Figure:
    """Timeline Plotly con linee di collegamento."""
    if not gq.identities:
        return go.Figure()

    rows: list[dict] = []
    for iid, members in gq.identities.items():
        for m in members:
            cam = m["camera_id"]
            date = m.get("date", "?")
            y_label = f"{cam} ({date})" if group_by_date else cam
            rows.append(
                {
                    "identity": iid,
                    "camera": cam,
                    "date": date,
                    "y_label": y_label,
                    "time": datetime.fromtimestamp(m["time"]),
                    "track_id": m["track_id"],
                    "video": m["video"],
                    "node_id": m["node_id"],
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["identity", "time"])

    fig = go.Figure()
    all_ids = sorted(gq.identities.keys())
    id_colors = {iid: f"hsl({(i * 137) % 360}, 75%, 55%)" for i, iid in enumerate(all_ids)}

    for iid in all_ids:
        sub = df[df["identity"] == iid]
        if sub.empty:
            continue
        is_highlight = highlight_identity is None or iid == highlight_identity
        color = id_colors[iid] if is_highlight else "#475569"
        opacity = 1.0 if is_highlight else 0.12
        width = 2.5 if is_highlight else 0.7
        size = 10 if is_highlight else 4

        if len(sub) > 1:
            fig.add_trace(
                go.Scatter(
                    x=sub["time"],
                    y=sub["y_label"],
                    mode="lines",
                    line={"color": color, "width": width, "dash": "solid"},
                    opacity=opacity,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        fig.add_trace(
            go.Scatter(
                x=sub["time"],
                y=sub["y_label"],
                mode="markers",
                marker={
                    "size": size,
                    "color": color,
                    "symbol": "diamond",
                    "line": {"width": 1.5, "color": _BG_DARK},
                },
                opacity=opacity,
                name=iid,
                customdata=sub[["track_id", "camera", "date", "video"]],
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Time: %{x}<br>"
                    "Y: %{y}<br>"
                    "Track: %{customdata[0]}<br>"
                    "Camera: %{customdata[1]}<br>"
                    "Date: %{customdata[2]}<br>"
                    "Video: %{customdata[3]}<br>"
                    "<extra></extra>"
                ),
            )
        )

    title = (
        f"<b>Timeline</b>  <span style='color:#94a3b8'>Evidenziato: {highlight_identity}</span>"
        if highlight_identity
        else "<b>Timeline</b>"
    )
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "color": _TEXT_MAIN}, "x": 0.5},
        height=550,
        xaxis_title={"text": "Orario", "font": {"color": _TEXT_MUTED, "size": 11}},
        yaxis_title={
            "text": "Camera" + (" + Data" if group_by_date else ""),
            "font": {"color": _TEXT_MUTED, "size": 11},
        },
        plot_bgcolor=_BG_DARK,
        paper_bgcolor=_BG_DARK,
        font={"color": _TEXT_MAIN, "family": "Segoe UI, sans-serif"},
        hovermode="closest",
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1,
            "xanchor": "left",
            "x": 1.02,
            "font": {"color": _TEXT_MUTED, "size": 9},
            "bgcolor": _BG_CARD,
        },
        xaxis={"gridcolor": _BORDER, "tickfont": {"color": _TEXT_MUTED, "size": 9}},
        yaxis={"gridcolor": _BORDER, "tickfont": {"color": _TEXT_MUTED, "size": 9}},
        margin={"t": 50, "b": 50, "l": 90, "r": 70},
    )
    return fig


# --- Bar chart attività ---


def render_camera_activity(gq: GraphQueries) -> Figure:
    """Bar chart orizzontale avvistamenti per camera."""
    activity = gq.get_camera_activity()
    if not activity:
        return go.Figure()

    cams = [a.camera_id for a in activity]
    identities = [a.total_identities for a in activity]
    tracklets = [a.total_tracklets for a in activity]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Identità",
            y=cams,
            x=identities,
            orientation="h",
            marker_color=_ACCENT,
            text=identities,
            textposition="outside",
            textfont={"color": _TEXT_MAIN, "size": 10},
        )
    )
    fig.add_trace(
        go.Bar(
            name="Tracklets",
            y=cams,
            x=tracklets,
            orientation="h",
            marker_color="#60a5fa",
            text=tracklets,
            textposition="outside",
            textfont={"color": _TEXT_MAIN, "size": 10},
        )
    )

    fig.update_layout(
        title={
            "text": "<b>Attività per Telecamera</b>",
            "font": {"size": 14, "color": _TEXT_MAIN},
            "x": 0.5,
        },
        barmode="group",
        xaxis_title={"text": "Conteggio", "font": {"color": _TEXT_MUTED, "size": 10}},
        yaxis_title={"text": "Camera", "font": {"color": _TEXT_MUTED, "size": 10}},
        height=max(300, len(cams) * 45),
        plot_bgcolor=_BG_DARK,
        paper_bgcolor=_BG_DARK,
        font={"color": _TEXT_MAIN, "family": "Segoe UI, sans-serif"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"color": _TEXT_MUTED, "size": 10},
        },
        xaxis={"gridcolor": _BORDER, "tickfont": {"color": _TEXT_MUTED, "size": 9}},
        yaxis={"tickfont": {"color": _TEXT_MUTED, "size": 9}},
        margin={"t": 50, "b": 40, "l": 70, "r": 40},
    )
    return fig


# --- Pie chart ---


def render_identity_distribution(gq: GraphQueries) -> Figure:
    """Pie chart singleton / intra / cross."""
    cross_stats = gq.get_cross_statistics()
    singletons = cross_stats["total_identities"] - cross_stats["multi_member_identities"]
    intra_only = cross_stats["multi_member_identities"] - cross_stats["cross_camera_identities"]
    cross_cam = cross_stats["cross_camera_identities"]

    labels = ["Singleton", "Intra-camera", "Cross-camera"]
    values = [singletons, intra_only, cross_cam]
    colors = ["#64748b", "#60a5fa", "#f472b6"]

    fig = go.Figure(
        data=go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker_colors=colors,
            textinfo="label+percent",
            textfont_size=13,
            textfont_color=_TEXT_MAIN,
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": "<b>Distribuzione Identità</b>",
            "font": {"size": 14, "color": _TEXT_MAIN},
            "x": 0.5,
        },
        height=380,
        width=380,
        plot_bgcolor=_BG_DARK,
        paper_bgcolor=_BG_DARK,
        font={"color": _TEXT_MAIN, "family": "Segoe UI, sans-serif"},
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.1,
            "xanchor": "center",
            "x": 0.5,
            "font": {"color": _TEXT_MUTED, "size": 10},
        },
        annotations=[
            {
                "text": f"<b>{cross_stats['total_identities']}</b>",
                "font": {"size": 20, "color": _ACCENT},
                "showarrow": False,
            },
            {
                "text": "identità",
                "font": {"size": 10, "color": _TEXT_MUTED},
                "showarrow": False,
                "yshift": -18,
            },
        ],
    )
    return fig


# --- Export helpers ---


def export_figure(
    fig: Figure, path: Path, width: int = 1200, height: int = 800, scale: int = 2
) -> None:
    """Esporta figura Plotly in PNG/SVG/PDF (richiede kaleido)."""
    fig.write_image(str(path), width=width, height=height, scale=scale)


def export_graph_html(html: str, path: Path) -> None:
    """Salva HTML del grafo pyvis su file."""
    path.write_text(html, encoding="utf-8")


# --- Stile tesi (export chiaro per stampa) ---

_THESIS_FONT = {"family": "Inter, sans-serif", "size": 14, "color": "#0f172a"}


def apply_thesis_style(fig: Figure) -> Figure:
    """Copia chiara di una figura Plotly: fondo bianco, testi scuri e leggibili."""
    thesis_fig = go.Figure(fig)
    thesis_fig.update_layout(
        template="plotly_white",
        font=dict(_THESIS_FONT),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return thesis_fig


def export_thesis_figure(
    fig: Figure, path: Path, width: int = 1400, height: int = 800, scale: int = 2
) -> None:
    """Esporta figura Plotly in versione tesi (richiede kaleido)."""
    export_figure(apply_thesis_style(fig), path, width=width, height=height, scale=scale)


def export_identity_strip(
    gq: GraphQueries,
    identity_id: str,
    roi_dir: Path,
    out_path: Path,
    thumb_height: int = 256,
    max_sightings: int = 12,
) -> Path | None:
    """Striscia PNG con un crop per avvistamento + label camera/ora.

    Ritorna `out_path` o None se nessuna ROI trovata.
    """
    from PIL import Image, ImageDraw

    members = gq.query_by_identity(identity_id)
    if not members:
        return None

    thumbs: list[tuple] = []
    for m in members[:max_sightings]:
        track_dir = roi_dir / m["video"] / f"Track_{m['track_id']:04d}"
        if not track_dir.exists():
            continue
        jpgs = sorted(track_dir.glob("*.jpg"))
        if not jpgs:
            continue
        img = Image.open(jpgs[len(jpgs) // 2]).convert("RGB")
        w = max(1, int(img.width * thumb_height / img.height))
        thumbs.append((img.resize((w, thumb_height), Image.LANCZOS), m))

    if not thumbs:
        return None

    label_h, gap = 24, 6
    strip = Image.new(
        "RGB",
        (sum(t.width for t, _ in thumbs) + gap * (len(thumbs) - 1), thumb_height + label_h),
        "white",
    )
    draw = ImageDraw.Draw(strip)
    x = 0
    for img, m in thumbs:
        strip.paste(img, (x, 0))
        ts = datetime.fromtimestamp(m["time"]).strftime("%H:%M:%S")
        draw.text((x + 4, thumb_height + 4), f"{m['camera_id']} {ts}", fill="black")
        x += img.width + gap

    strip.save(out_path)
    return out_path
