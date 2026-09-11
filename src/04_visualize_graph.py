"""Dashboard Streamlit per esplorare il grafo di movimento globale.

Tabs:
- Overview: grafo interattivo, timeline, sequenza ROI.
- Statistiche: heatmap, distribuzione identità, attività per camera.
- Query: ricerca per percorso camera o per identità.

Usage:
    streamlit run src/04_visualize_graph.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path  # noqa: TC003

import pandas as pd
import streamlit as st

from src.config import PROJECT_ROOT
from src.dashboard_viz import (
    export_figure,
    export_graph_html,
    export_identity_strip,
    export_thesis_figure,
    render_camera_activity,
    render_camera_graph,
    render_heatmap,
    render_identity_distribution,
    render_sankey,
    render_timeline,
)
from src.graph_queries import GraphQueries

# --- Configurazione pagina ---

st.set_page_config(
    page_title="Ariadne — Multi-Camera Re-ID Dashboard",
    page_icon="🕸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS glassmorphism ---

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 41, 59, 0.6);
        padding: 8px 12px;
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 12px;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 0.9rem;
        color: #94a3b8;
        background: rgba(51, 65, 85, 0.5);
        border: 1px solid rgba(99, 102, 241, 0.15);
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: #ffffff !important;
        border-color: transparent !important;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.2);
    }
    .metric-card h2 {
        margin: 0;
        background: linear-gradient(135deg, #6366f1 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-card p {
        margin: 6px 0 0;
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 500;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .section-title {
        color: #f8fafc;
        font-size: 1.25rem;
        font-weight: 700;
        margin: 24px 0 8px;
        letter-spacing: -0.3px;
    }
    .section-caption {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-bottom: 12px;
        line-height: 1.5;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-right: 6px;
    }
    .badge-cross-cam {
        background: linear-gradient(135deg, #f59e0b 0%, #f97316 100%);
        color: white;
    }
    .badge-cross-day {
        background: linear-gradient(135deg, #ec4899 0%, #db2777 100%);
        color: white;
    }
    .badge-both {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
    }
    .arrow-transition {
        text-align: center;
        padding: 20px 4px;
    }
    .arrow-transition span {
        font-size: 28px;
    }
    .arrow-transition .gap-label {
        font-size: 11px;
        color: #94a3b8;
        margin-top: 4px;
    }
    .info-box {
        background: rgba(99, 102, 241, 0.1);
        border-left: 3px solid #6366f1;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 12px 0;
        color: #c4b5fd;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Path ---

OUTPUT_DIR = PROJECT_ROOT / "output"
ROI_DIR = PROJECT_ROOT / "data" / "processed" / "extracted_rois"


# --- Helper ROI ---


def _get_roi_image_paths(video: str, track_id: int, base_dir: Path = ROI_DIR) -> list[Path]:
    """Path dei JPEG di una tracklet."""
    track_dir = base_dir / video / f"Track_{track_id:04d}"
    if not track_dir.exists():
        return []
    return sorted(track_dir.glob("*.jpg"))


def _roi_gallery(video: str, track_id: int, max_preview: int = 6) -> None:
    """Gallery dei frame della tracklet."""
    paths = _get_roi_image_paths(video, track_id)
    if not paths:
        st.caption("🖼️ Nessuna immagine trovata")
        return

    if len(paths) <= max_preview:
        preview = paths
    else:
        step = max(1, len(paths) // max_preview)
        preview = [paths[i] for i in range(0, len(paths), step)][:max_preview]

    st.caption(f"📸 **{len(paths)}** frame — anteprima:")
    cols = st.columns(len(preview))
    for idx, p in enumerate(preview):
        with cols[idx]:
            st.image(str(p), width=150)

    if len(paths) > max_preview:
        with st.expander(f"📁 Espandi tutti i {len(paths)} frame"):
            batch = 6
            for start in range(0, len(paths), batch):
                batch_paths = paths[start : start + batch]
                st.image([str(p) for p in batch_paths], width=110)


def _roi_sequence(members: list[dict], per_page: int = 6, label: str = "Avvistamenti") -> None:
    """Mostra ROI in sequenza orizzontale con frecce e gap temporali.

    Oltre `per_page` avvistamenti, il resto va in sezioni espandibili.
    """
    if not members:
        st.info("Nessuna ROI disponibile.")
        return

    def _show_chunk(chunk: list[dict]) -> None:
        n = len(chunk)
        # Layout alternato: [ROI][freccia][ROI]... con colonne di larghezza diversa
        ratios = [1] + [0.4, 1] * (n - 1)
        cols = st.columns(ratios)
        idx = 0

        for i, m in enumerate(chunk):
            ts = datetime.fromtimestamp(m["time"]).strftime("%H:%M:%S")
            date = datetime.fromtimestamp(m["time"]).strftime("%Y-%m-%d")

            with cols[idx]:
                st.markdown(f"**{m['camera_id']}** | `{date} {ts}`")
                _roi_gallery(m["video"], m["track_id"], max_preview=4)
            idx += 1

            if i < n - 1:
                gap = chunk[i + 1]["time"] - m["time"]
                with cols[idx]:
                    st.markdown(
                        f"<div class='arrow-transition'>"
                        f"<span>➡️</span><br>"
                        f"<span class='gap-label'>{gap:.0f}s</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                idx += 1

    chunks = [members[i : i + per_page] for i in range(0, len(members), per_page)]
    _show_chunk(chunks[0])
    for k, chunk in enumerate(chunks[1:], start=1):
        start_n = k * per_page + 1
        end_n = min((k + 1) * per_page, len(members))
        with st.expander(f"{label} {start_n}–{end_n} di {len(members)}"):
            _show_chunk(chunk)

    st.markdown(
        "<div class='info-box'>"
        "ℹ️ <b>Gap</b> = tempo tra due avvistamenti consecutivi. "
        "Non è velocità di percorrenza: la persona potrebbe aver percorso "
        "aree non coperte dalle telecamere."
        "</div>",
        unsafe_allow_html=True,
    )


# --- Caricamento grafo ---


def _init_graph() -> tuple[GraphQueries, dict]:
    path = OUTPUT_DIR / "global_graph.json"
    if not path.exists():
        st.error("❌ Grafo non trovato. Esegui prima il Modulo 3 (build_graph).")
        st.stop()
        return GraphQueries(path), {}  # pyright: ignore[reportUnreachable]
    gq = GraphQueries(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return gq, raw


graph_queries, raw_graph = _init_graph()

# --- Sidebar ---

st.sidebar.markdown(
    """
    <div style="text-align:center;margin-bottom:16px;">
        <h1 style="background:linear-gradient(135deg,#6366f1,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:1.8rem;margin:0;">🕸 Ariadne</h1>
        <p style="color:#94a3b8;font-size:0.8rem;margin:4px 0 0;">Multi-Camera Re-ID v3.0</p>
        <p style="color:#6366f1;font-size:0.7rem;margin:2px 0 0;">Clustering Gerarchico 3-Fasi</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

all_ids = sorted(graph_queries.identities.keys())

if "sb_identity" not in st.session_state:
    st.session_state["sb_identity"] = "Tutte"
if "highlight_path" not in st.session_state:
    st.session_state["highlight_path"] = None

selected_identity = st.sidebar.selectbox(
    "🎯 Seleziona persona",
    ["Tutte", *all_ids],
    key="sb_identity",
    help="L'identità viene evidenziata in Grafo, Timeline e ROI",
)
highlight_id = None if selected_identity == "Tutte" else selected_identity
highlight_path = st.session_state.get("highlight_path")

st.sidebar.markdown("---")

# Metriche globali
metrics = graph_queries.get_global_metrics()

st.sidebar.subheader("📊 Metriche Globali")
for label, value in [
    ("Identità", metrics["num_identities"]),
    ("Cross-camera", metrics["cross_camera_identities"]),
    ("Cross-day", metrics["cross_day_identities"]),
    ("Archi", metrics["num_edges"]),
]:
    st.sidebar.metric(label, value)

with st.sidebar.expander("Dettagli avanzati"):
    st.sidebar.metric("Cross entrambi", metrics["cross_camera_and_day"])
    st.sidebar.metric("Ratio cross-cam", f"{metrics['cross_camera_ratio']:.1%}")
    if "clustering" in raw_graph:
        st.sidebar.markdown(f"**Clustering:** `{raw_graph['clustering']}`")
    if "threshold_intra" in raw_graph:
        st.sidebar.markdown(
            f"**Soglie:** intra={raw_graph['threshold_intra']}, "
            f"cross-cam={raw_graph['threshold_cross_cam']}, "
            f"cross-day={raw_graph['threshold_cross_day']}"
        )

st.sidebar.markdown("---")

# --- Tabs principali ---

st.markdown(
    """
    <div style="text-align:center;margin-bottom:8px;">
        <h1 style="background:linear-gradient(135deg,#6366f1,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:2rem;margin:0;">🕸 Ariadne</h1>
        <p style="color:#94a3b8;font-size:0.9rem;margin:6px 0 0;">
            Multi-Camera Person Re-Identification & Trajectory Analysis
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

tab_overview, tab_stats, tab_query, tab_compare = st.tabs(
    ["📋 Overview", "📊 Statistiche", "🔍 Query", "🆚 Confronto"]
)

# --- Tab Overview ---

with tab_overview:
    st.markdown(
        "<div class='section-title'>🎨 Grafo delle Telecamere</div>", unsafe_allow_html=True
    )

    group_by_date = st.checkbox(
        "📅 Raggruppa per data",
        value=False,
        help="Ogni telecamera è divisa per giornata (utile per verificare cross-day).",
    )

    if highlight_path:
        st.markdown(
            f"<div class='info-box'>"
            f"🎯 Percorso evidenziato: <b>{' → '.join(highlight_path)}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if st.button("❌ Rimuovi evidenziazione percorso", key="reset_path"):
            st.session_state["highlight_path"] = None
            st.rerun()

    if group_by_date:
        st.markdown(
            "<div class='section-caption'>"
            "Nodi = telecamera + data. Archi = transizioni. "
            "Bordo <b style='color:#f59e0b'>arancione</b> = percorso persona selezionata."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='section-caption'>"
            "Nodi = telecamere (collassate per data). Archi = transizioni. "
            "Bordo <b style='color:#f59e0b'>arancione</b> = percorso persona selezionata."
            "</div>",
            unsafe_allow_html=True,
        )

    html_graph = render_camera_graph(
        graph_queries,
        highlight_identity=highlight_id,
        highlight_path=highlight_path,
        min_edge_weight=1,
        physics=True,
        group_by_date=group_by_date,
    )
    st.components.v1.html(html_graph, height=550)
    if st.button("📷 Esporta grafo HTML", key="exp_graph"):
        out = OUTPUT_DIR / "camera_graph.html"
        export_graph_html(html_graph, out)
        st.success(f"Salvato: {out}")

    st.markdown("---")
    st.markdown("<div class='section-title'>🕒 Timeline</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-caption'>"
        "Marker = avvistamento. Linee = percorso della persona. Grigio = altre persone."
        "</div>",
        unsafe_allow_html=True,
    )

    fig_tl = render_timeline(
        graph_queries, highlight_identity=highlight_id, group_by_date=group_by_date
    )
    st.plotly_chart(fig_tl, use_container_width=True)
    if st.button("📷 Esporta timeline (tesi)", key="exp_tl"):
        out = OUTPUT_DIR / "timeline_thesis.png"
        export_thesis_figure(fig_tl, out, width=1400, height=700)
        st.success(f"Salvata: {out}")

    st.markdown("---")
    st.markdown("<div class='section-title'>👤 Sequenza ROI</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-caption'>"
        "Immagini estratte della persona. Frecce = transizione con gap temporale."
        "</div>",
        unsafe_allow_html=True,
    )

    if highlight_id:
        members = graph_queries.query_by_identity(highlight_id)
        _roi_sequence(members, label=f"Panoramica {highlight_id}")
        if st.button("📷 Esporta strip ROI (tesi)", key="exp_roi_over"):
            out = OUTPUT_DIR / f"roi_{highlight_id}.png"
            if export_identity_strip(graph_queries, highlight_id, ROI_DIR, out):
                st.success(f"Salvata: {out}")
            else:
                st.warning("Nessuna ROI trovata.")
    else:
        st.info("Seleziona un'identità dalla sidebar per visualizzare la sequenza delle ROI.")

# --- Tab Statistiche ---

with tab_stats:
    st.markdown("<div class='section-title'>📊 Panoramica Statistica</div>", unsafe_allow_html=True)

    # Card metriche
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, value in [
        (c1, "Identità", metrics["num_identities"]),
        (c2, "Cross-camera", metrics["cross_camera_identities"]),
        (c3, "Cross-day", metrics["cross_day_identities"]),
        (c4, "Archi", metrics["num_edges"]),
        (c5, "Nodi", metrics["num_nodes"]),
    ]:
        col.markdown(
            f"""
            <div class="metric-card">
                <h2>{value}</h2>
                <p>{label}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Heatmap
    st.markdown("<div class='section-title'>🔥 Heatmap Transizioni</div>", unsafe_allow_html=True)
    col_hm, col_hm_opts = st.columns([5, 1])
    with col_hm_opts:
        use_gap = st.checkbox("Colora per gap", value=False, help="Colora per gap temporale medio")
        if st.button("📷 Esporta", key="exp_hm"):
            out = OUTPUT_DIR / "heatmap.png"
            export_figure(
                render_heatmap(graph_queries, use_avg_gap=use_gap),
                out,
                width=1200,
                height=900,
                scale=2,
            )
            st.success(f"Salvata: {out}")
    with col_hm:
        st.plotly_chart(
            render_heatmap(graph_queries, use_avg_gap=use_gap), use_container_width=True
        )

    st.markdown("---")
    st.markdown("<div class='section-title'>🌊 Flussi tra Camere</div>", unsafe_allow_html=True)
    fig_sk = render_sankey(graph_queries)
    st.plotly_chart(fig_sk, use_container_width=True)
    if st.button("📷 Esporta Sankey (tesi)", key="exp_sk"):
        out = OUTPUT_DIR / "sankey_thesis.png"
        export_thesis_figure(fig_sk, out, width=1400, height=700)
        st.success(f"Salvata: {out}")

    # Pie + Bar
    col_pie, col_bar = st.columns(2)
    with col_pie:
        st.markdown(
            "<div class='section-title'>🎯 Distribuzione Identità</div>", unsafe_allow_html=True
        )
        st.plotly_chart(render_identity_distribution(graph_queries), use_container_width=True)
        if st.button("📷 Esporta", key="exp_pie"):
            out = OUTPUT_DIR / "pie.png"
            export_figure(
                render_identity_distribution(graph_queries), out, width=500, height=500, scale=2
            )
            st.success(f"Salvata: {out}")

    with col_bar:
        st.markdown(
            "<div class='section-title'>📹 Attività per Camera</div>", unsafe_allow_html=True
        )
        st.plotly_chart(render_camera_activity(graph_queries), use_container_width=True)
        if st.button("📷 Esporta", key="exp_bar"):
            out = OUTPUT_DIR / "bar.png"
            export_figure(
                render_camera_activity(graph_queries), out, width=800, height=600, scale=2
            )
            st.success(f"Salvata: {out}")

    # Top percorsi
    st.markdown("---")
    st.markdown(
        "<div class='section-title'>🏆 Top Percorsi Cross-camera</div>", unsafe_allow_html=True
    )
    top_paths = graph_queries.get_path_statistics(top_k=10)
    if top_paths:
        st.dataframe(
            pd.DataFrame(top_paths),
            column_config={
                "path": st.column_config.TextColumn("Percorso", width="large"),
                "count": st.column_config.NumberColumn("Occ."),
                "avg_duration_sec": st.column_config.NumberColumn("Durata (s)"),
                "num_cameras": st.column_config.NumberColumn("Cam"),
            },
            hide_index=True,
        )
    else:
        st.info("Nessun percorso multi-camera.")

    # Transizioni dettagliate
    st.markdown("---")
    st.markdown(
        "<div class='section-title'>🔄 Transizioni Dettagliate</div>", unsafe_allow_html=True
    )
    trans = graph_queries.get_transition_statistics()
    if trans:
        st.dataframe(
            pd.DataFrame([t.__dict__ for t in trans]),
            column_config={
                "src_camera": st.column_config.TextColumn("Da"),
                "dst_camera": st.column_config.TextColumn("A"),
                "count": st.column_config.NumberColumn("N"),
                "avg_time_gap": st.column_config.NumberColumn("Gap avg (s)"),
                "cross_day_count": st.column_config.NumberColumn("Cross-day"),
            },
            hide_index=True,
        )
    else:
        st.info("Nessuna transizione.")

# --- Tab Query ---

with tab_query:
    st.markdown(
        "<div class='section-title'>🔍 Query Percorsi e Identità</div>", unsafe_allow_html=True
    )
    st.markdown(
        "<div class='section-caption'>"
        "Dato un <b>percorso</b> → trova le persone. Dato una <b>persona</b> → trova il percorso."
        "</div>",
        unsafe_allow_html=True,
    )

    q1, q2 = st.columns(2)

    with q1:
        st.markdown("##### 🗺️ Cerca per Percorso")
        path_input = st.text_input(
            "Sequenza camere",
            placeholder="G505, G507, G639",
            help="Separa le camere con virgola",
        )
        date_filter = st.text_input(
            "Data (opzionale)",
            placeholder="2018-05-18",
            help="Lascia vuoto per tutte le date",
        )
        allow_sub = st.checkbox("Sotto-sequenza", value=True)

        if st.button("🔍 Cerca Percorso", key="btn_path"):
            pattern = [c.strip() for c in path_input.split(",") if c.strip()]
            if not pattern:
                st.warning("Inserisci almeno una camera.")
            else:
                results = graph_queries.query_by_path(
                    pattern,
                    allow_subsequence=allow_sub,
                    date_filter=date_filter or None,
                )
                if not results:
                    st.info("Nessuna identità trovata.")
                else:
                    st.success(f"Trovate **{len(results)}** identità")
                    for i, r in enumerate(results):
                        badges = ""
                        if r.is_cross_camera and r.is_cross_day:
                            badges = '<span class="badge badge-both">Cross-Cam + Cross-Day</span>'
                        elif r.is_cross_camera:
                            badges = '<span class="badge badge-cross-cam">Cross-Camera</span>'
                        elif r.is_cross_day:
                            badges = '<span class="badge badge-cross-day">Cross-Day</span>'

                        with st.expander(f"{r.identity_id} — {' → '.join(r.camera_sequence)}"):
                            st.markdown(badges, unsafe_allow_html=True)
                            st.write(f"**Camere:** {', '.join(r.camera_sequence)}")
                            st.write(f"**Date:** {', '.join(sorted(set(r.dates)))}")
                            st.write(
                                f"**Orari:** {[datetime.fromtimestamp(t).strftime('%H:%M:%S') for t in r.times]}"
                            )
                            if st.button("🎯 Evidenzia nel grafo", key=f"hl_path_{i}"):
                                st.session_state["highlight_path"] = r.camera_sequence
                                st.session_state["sb_identity"] = "Tutte"
                                st.rerun()
                            st.markdown(
                                "<div class='info-box'>"
                                "ℹ️ <b>Gap</b> = tempo tra avvistamenti consecutivi. "
                                "Non è la velocità di percorrenza."
                                "</div>",
                                unsafe_allow_html=True,
                            )
                            members = graph_queries.query_by_identity(r.identity_id)
                            st.markdown("**👤 ROI:**")
                            _roi_sequence(members, label=f"Percorso {r.identity_id}")

    with q2:
        st.markdown("##### 👤 Cerca per Identità")
        sel_id = st.selectbox("Seleziona identità", all_ids, index=0)

        if sel_id:
            summary = graph_queries.get_identity_summary(sel_id)
            if summary:
                badges = ""
                if summary["is_cross_camera"] and summary["is_cross_day"]:
                    badges = '<span class="badge badge-both">Cross-Cam + Cross-Day</span>'
                elif summary["is_cross_camera"]:
                    badges = '<span class="badge badge-cross-cam">Cross-Camera</span>'
                elif summary["is_cross_day"]:
                    badges = '<span class="badge badge-cross-day">Cross-Day</span>'
                else:
                    badges = '<span class="badge">Single-Camera</span>'

                st.markdown(badges, unsafe_allow_html=True)
                st.markdown("**📋 Riepilogo**")
                st.write(f"**Avvistamenti:** {summary['num_sightings']}")
                st.write(f"**Camere uniche:** {summary['num_unique_cameras']}")
                st.write(f"**Giorni:** {summary['num_unique_days']}")
                st.write(f"**Sequenza:** {' → '.join(summary['cameras'])}")
                st.write(f"**Durata totale:** {summary['duration_sec']:.1f}s")

                members = graph_queries.query_by_identity(sel_id)
                st.markdown("**🕒 Dettaglio avvistamenti**")
                for m in members:
                    ts = datetime.fromtimestamp(m["time"]).strftime("%Y-%m-%d %H:%M:%S")
                    st.markdown(f"- `{ts}` | **{m['camera_id']}** | Track {m['track_id']}")

                st.markdown("**👤 ROI:**")
                _roi_sequence(members, label=f"Identità {sel_id}")
                if st.button("📷 Esporta strip ROI (tesi)", key="exp_roi_id"):
                    out = OUTPUT_DIR / f"roi_{sel_id}.png"
                    if export_identity_strip(graph_queries, sel_id, ROI_DIR, out):
                        st.success(f"Salvata: {out}")
                    else:
                        st.warning("Nessuna ROI trovata.")

                if st.button("🎯 Evidenzia in Overview", key="btn_hl"):
                    st.session_state["sb_identity"] = sel_id
                    st.rerun()

# --- Tab Confronto ---

with tab_compare:
    st.markdown(
        "<div class='section-title'>🆚 Confronto tra due run</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='section-caption'>"
        "Seleziona due grafi generati con parametri diversi per confrontarne le statistiche."
        "</div>",
        unsafe_allow_html=True,
    )

    graph_files = sorted(OUTPUT_DIR.glob("global_graph*.json"))
    if len(graph_files) < 2:
        st.info("Servono almeno due file global_graph*.json in output/ per il confronto.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            file_a = st.selectbox(
                "Run A",
                graph_files,
                format_func=lambda p: p.name,
                key="cmp_file_a",
            )
        with col_b:
            file_b = st.selectbox(
                "Run B",
                [f for f in graph_files if f != file_a],
                format_func=lambda p: p.name,
                key="cmp_file_b",
            )

        if file_a and file_b:
            gq_a = GraphQueries(file_a)
            gq_b = GraphQueries(file_b)
            m_a = gq_a.get_global_metrics()
            m_b = gq_b.get_global_metrics()

            st.markdown("##### Metriche globali")
            metric_cols = [
                ("Identità", "num_identities"),
                ("Tracklet", "num_tracklets"),
                ("Archi", "num_edges"),
                ("Nodi", "num_nodes"),
                ("Cross-camera", "cross_camera_identities"),
                ("Cross-day", "cross_day_identities"),
            ]
            cols = st.columns(len(metric_cols))
            for col, (label, key) in zip(cols, metric_cols, strict=True):
                va = m_a.get(key, 0)
                vb = m_b.get(key, 0)
                col.metric(label, vb, f"{vb - va:+,}", delta_color="off")

            st.markdown("##### Top percorsi")
            top_a = gq_a.get_path_statistics(top_k=10)
            top_b = gq_b.get_path_statistics(top_k=10)
            ca, cb = st.columns(2)
            with ca:
                st.caption(f"**{file_a.name}**")
                if top_a:
                    st.dataframe(
                        pd.DataFrame(top_a),
                        column_config={
                            "path": st.column_config.TextColumn("Percorso", width="large"),
                            "count": st.column_config.NumberColumn("Occ."),
                            "num_cameras": st.column_config.NumberColumn("Cam"),
                        },
                        hide_index=True,
                    )
                else:
                    st.info("Nessun percorso multi-camera.")
            with cb:
                st.caption(f"**{file_b.name}**")
                if top_b:
                    st.dataframe(
                        pd.DataFrame(top_b),
                        column_config={
                            "path": st.column_config.TextColumn("Percorso", width="large"),
                            "count": st.column_config.NumberColumn("Occ."),
                            "num_cameras": st.column_config.NumberColumn("Cam"),
                        },
                        hide_index=True,
                    )
                else:
                    st.info("Nessun percorso multi-camera.")

st.markdown("---")
st.caption("Ariadne v3.0 — Cross-camera Person Re-identification")
