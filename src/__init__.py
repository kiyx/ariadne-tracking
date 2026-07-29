"""Ariadne Tracking — src package.

Questo pacchetto contiene i moduli principali della pipeline:

- config: costanti, path e parametri di default
- utils: utility condivise (logging, video, Re-ID, clustering)
- 01_tracker_extractor: detection + tracking + estrazione ROI
- 02_feature_extractor: estrazione embedding con C2DResNet50 CAL
- 03_build_graph: clustering gerarchico e costruzione grafo globale
- 04_visualize_graph: dashboard Streamlit per esplorare i grafi
- dashboard_viz: visualizzazioni Plotly/pyvis indipendenti da Streamlit
- graph_queries: query e statistiche sui grafi
- eval_mevid: valutazione Re-ID sul benchmark MEVID
"""
