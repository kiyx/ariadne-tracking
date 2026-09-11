<div align="center">

# Ariadne Tracking

**Dalla re-identificazione alla rappresentazione a grafo: modellazione dei percorsi umani in scenari multi-camera.**

Pipeline modulare per il tracking multi-camera di persone e la costruzione di un grafo di movimento globale, robusta ai cambi d'abito e alle discontinuità temporali.

[![CI](https://github.com/kiyx/ariadne-tracking/actions/workflows/ci.yml/badge.svg)](https://github.com/kiyx/ariadne-tracking/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26-111F68)](https://docs.ultralytics.com/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230)](https://github.com/astral-sh/ruff)

</div>

---

## Panoramica

**Ariadne Tracking** elabora video di sorveglianza provenienti da più telecamere e ricostruisce il percorso delle persone nello spazio e nel tempo. La pipeline:

1. **Rileva e traccia** le persone in ogni video (YOLO26 + BoT-SORT), estraendo ROI pulite per ogni tracklet;
2. **Estrae embedding Re-ID** robusti ai cambi d'abito con un modello *Clothes-Agnostic* (C2DResNet50 + CAL);
3. **Costruisce un grafo di movimento globale** che collega le tracklet della stessa identità anche tra telecamere e giorni diversi;
4. **Visualizza i percorsi** in una dashboard interattiva con grafi, timeline, heatmap e query.

A differenza di un semplice modulo di tracking, il progetto modella l'intero flusso *detection → identità → movimento*, ed è pensato per essere **scene-agnostic**: ogni coppia camera/giorno è un nodo distinto del grafo, così le identità non vengono mai confuse tra contesti diversi.

> **Tesi di Laurea Triennale in Informatica** — Università degli Studi di Napoli Federico II
>
> | | |
> | :--- | :--- |
> | **Autore** | Giuseppe Paolo Esposito |
> | **Relatore** | Prof. Daniel Riccio |
> | **Anno Accademico** | 2025/2026 |

---

## Pipeline

```text
                    Modulo 0                    Modulo 1
               ┌──────────────┐    ┌───────────────────────────┐
 Video MEVID ──┤  S3 Download ├───►│ YOLO26mpose + BoT-SORT    │
               └──────────────┘    │ Detection → Tracking → ROI│
                                   └─────────────┬─────────────┘
                                                 │
                    Modulo 2                     ▼
               ┌──────────────────────────────────────┐
               │ C2DResNet50 + CAL → Embedding 2048-D │
               └──────────────┬───────────────────────┘
                              │
                    Modulo 3  ▼
               ┌──────────────────────────────────────┐
               │ Grafo di Movimento Globale (JSON)    │
               └──────────────┬───────────────────────┘
                              │
                    Modulo 4  ▼
               ┌──────────────────────────────────────┐
               │ Dashboard Streamlit + PyVis/Plotly   │
               └──────────────────────────────────────┘
```

| Modulo | File | Responsabilità |
| :--- | :--- | :--- |
| 0 — Dataset | `src/00_setup_dataset.py` | Download dei video MEVID da S3 (full o subset) |
| 1 — Tracking | `src/01_tracker_extractor.py` | Detection, tracking e filtri di qualità (pose, sfocatura, sovrapposizioni) → ROI |
| 2 — Re-ID | `src/02_feature_extractor.py` | Estrazione embedding 2048-D con C2DResNet50 + CAL |
| 3 — Grafo | `src/03_build_graph.py` | Clustering gerarchico a 3 fasi (intra-camera, cross-camera, cross-day) → grafo JSON |
| 4 — Visualizzazione | `src/04_visualize_graph.py` | Dashboard interattiva per esplorare grafi, statistiche e percorsi |

---

## Caratteristiche principali

- **Tracking multi-camera** con BoT-SORT e detection pose-guided, con filtri geometrici e di qualità per eliminare detection parziali o sfocate.
- **Re-Identification clothes-agnostic**: embedding basati su modello CAL, addestrato a ignorare l'abbigliamento — pensato per sorveglianza reale su più giorni.
- **Grafo di movimento globale** costruito con clustering gerarchico a soglie decrescenti: alta confidenza nello stesso contesto, più permissiva tra giorni diversi.
- **Dashboard interattiva** con grafo PyVis, timeline, heatmap, Sankey e query per identità o percorso camera.
- **Benchmark ufficiale MEVID** con protocollo standard (CMC + mAP).
- **Qualità del codice**: linting e formattazione con Ruff, type checking e test automatici in CI su ogni push/PR.

---

## Risultati

Valutazione sul benchmark **MEVID** (Multi-view Extended Videos with Identities, WACV 2023), split di test ufficiale (54 identità, 1 754 tracklet, 316 query):

| Metrica | Valore |
| :--- | ---: |
| Rank-1 | **52.53%** |
| Rank-5 | 66.77% |
| Rank-10 | 72.78% |
| Rank-20 | 80.70% |
| mAP | **27.01%** |

---

## Stack tecnologico

| Ambito | Tecnologie |
| :--- | :--- |
| Detection | YOLO26m-pose (Ultralytics) |
| Tracking | BoT-SORT |
| Re-Identification | C2DResNet50 + CAL (Clothes-Agnostic Learning), PyTorch / TorchVision |
| Clustering | Clustering agglomerativo gerarchico (SciPy) |
| Accelerazione | CUDA, TensorRT FP16 |
| Visualizzazione | Streamlit, PyVis, Plotly, NetworkX, Pandas |
| Qualità | Ruff, mypy, Pyright, pytest, GitHub Actions |

---

## Quick Start

**Requisiti:** Python ≥ 3.10, GPU NVIDIA con CUDA ≥ 12.x (testato su RTX 3060 6 GB), Conda.

```bash
# 1. Clona il repository
git clone https://github.com/kiyx/ariadne-tracking.git
cd ariadne-tracking

# 2. Crea l'ambiente
conda create -n ariadne_env python=3.10 -y
conda activate ariadne_env

# 3. PyTorch con CUDA + dipendenze
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 4. (Opzionale) Dashboard e tool di sviluppo
pip install -e ".[viz,dev]"
```

**Esegui la pipeline completa:**

```bash
python -m src.00_setup_dataset --subset   # download subset MEVID (~390 MB)
python -m src.01_tracker_extractor        # detection + tracking → ROI
python -m src.02_feature_extractor        # embedding Re-ID 2048-D
python -m src.03_build_graph              # grafo cross-camera → output/global_graph.json
streamlit run src/04_visualize_graph.py   # dashboard → http://localhost:8501
```

Ogni modulo espone opzioni CLI (soglie di similarità, batch size, TensorRT, limiti sui video, ecc.) documentate in `--help`. I parametri di default sono centralizzati in [`src/config.py`](src/config.py).

---

## Struttura del progetto

```text
ariadne-tracking/
├── src/
│   ├── 00_setup_dataset.py      # Download video MEVID da S3
│   ├── 01_tracker_extractor.py  # Detection + tracking → ROI extraction
│   ├── 02_feature_extractor.py  # C2DResNet50 + CAL → embedding 2048-D
│   ├── 03_build_graph.py        # Clustering gerarchico 3-fasi → grafo JSON
│   ├── 04_visualize_graph.py    # Dashboard interattiva (Streamlit)
│   ├── dashboard_viz.py         # Visualizzazioni Plotly/PyVis
│   ├── graph_queries.py         # Query e statistiche sui grafi
│   ├── eval_mevid.py            # Benchmark MEVID ufficiale (CMC + mAP)
│   └── config.py                # Costanti e parametri centralizzati
├── models/                      # Pesi YOLO e CAL + sorgente Simple-CCReID
├── data/                        # Video grezzi, ROI estratte, annotazioni MEVID
├── output/                      # Grafi, report e log
├── tests/                       # Test suite pytest
└── .github/workflows/           # CI: lint + type check + test
```

---

## Dataset

**MEVID** (Multi-view Extended Videos with Identities) è un dataset multi-camera di sorveglianza su più giorni, con **158 identità**, **598 outfit**, **8 092 tracklet** e **33 telecamere**. Il progetto include il download automatico dei video e uno script di valutazione che riproduce il protocollo ufficiale del benchmark.

---

## Sviluppo e qualità

```bash
ruff check .        # linting
ruff format .       # formattazione
mypy src/           # type checking
pytest              # test suite
```

La CI (GitHub Actions) esegue automaticamente lint, format check, type checking e test a ogni push e pull request sul branch `main`.

---

## Licenza

Progetto di tesi — Università degli Studi di Napoli Federico II.
