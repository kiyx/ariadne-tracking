<div align="center">

**English** | [Italiano](README.it.md)

# Ariadne Tracking

**From Person Re-Identification to Graph Representation: Modeling Human Paths in Multi-Camera Scenarios.**

A modular pipeline for multi-camera person tracking and global movement graph construction, robust to clothing changes and temporal discontinuities.

[![CI](https://github.com/kiyx/ariadne-tracking/actions/workflows/ci.yml/badge.svg)](https://github.com/kiyx/ariadne-tracking/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26-111F68)](https://docs.ultralytics.com/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230)](https://github.com/astral-sh/ruff)

</div>

---

## Overview

**Ariadne Tracking** processes surveillance videos from multiple cameras and reconstructs people's trajectories across space and time. The pipeline:

1. **Detects and tracks** people in each video (YOLO26 + BoT-SORT), extracting clean ROIs for every tracklet;
2. **Extracts Re-ID embeddings** robust to clothing changes with a *Clothes-Agnostic* model (C2DResNet50 + CAL);
3. **Builds a global movement graph** that links tracklets of the same identity across cameras and even different days;
4. **Visualizes trajectories** in an interactive dashboard with graphs, timelines, heatmaps and queries.

Unlike a plain tracking module, the project models the whole *detection → identity → movement* flow and is designed to be **scene-agnostic**: each camera/day pair is a distinct graph node, so identities are never confused across different contexts.

> **Bachelor's Degree Thesis in Computer Science** — University of Naples Federico II
>
> | | |
> | :--- | :--- |
> | **Author** | Giuseppe Paolo Esposito |
> | **Supervisor** | Prof. Daniel Riccio |
> | **Academic Year** | 2025/2026 |

---

## Pipeline

```text
                    Module 0                    Module 1
               ┌──────────────┐    ┌───────────────────────────┐
 MEVID video ──┤  S3 Download ├───►│ YOLO26mpose + BoT-SORT    │
               └──────────────┘    │ Detection → Tracking → ROI│
                                   └─────────────┬─────────────┘
                                                 │
                    Module 2                     ▼
               ┌──────────────────────────────────────┐
               │ C2DResNet50 + CAL → 2048-D Embedding │
               └──────────────┬───────────────────────┘
                              │
                    Module 3  ▼
               ┌──────────────────────────────────────┐
               │ Global Movement Graph (JSON)         │
               └──────────────┬───────────────────────┘
                              │
                    Module 4  ▼
               ┌──────────────────────────────────────┐
               │ Streamlit Dashboard + PyVis/Plotly   │
               └──────────────────────────────────────┘
```

| Module | File | Responsibility |
| :--- | :--- | :--- |
| 0 — Dataset | `src/00_setup_dataset.py` | MEVID video download from S3 (full or subset) |
| 1 — Tracking | `src/01_tracker_extractor.py` | Detection, tracking and quality filters (pose, blur, overlaps) → ROIs |
| 2 — Re-ID | `src/02_feature_extractor.py` | 2048-D embedding extraction with C2DResNet50 + CAL |
| 3 — Graph | `src/03_build_graph.py` | 3-phase hierarchical clustering (intra-camera, cross-camera, cross-day) → JSON graph |
| 4 — Visualization | `src/04_visualize_graph.py` | Interactive dashboard to explore graphs, statistics and trajectories |

---

## Key Features

- **Multi-camera tracking** with BoT-SORT and pose-guided detection, plus geometric and quality filters to drop partial or blurry detections.
- **Clothes-agnostic Re-Identification**: embeddings from the CAL model, trained to ignore clothing — designed for real-world multi-day surveillance.
- **Global movement graph** built with hierarchical clustering at decreasing thresholds: high confidence within the same context, more permissive across days.
- **Interactive dashboard** with PyVis graph, timeline, heatmap, Sankey and queries by identity or camera path.
- **Official MEVID benchmark** with the standard protocol (CMC + mAP).
- **Code quality**: linting and formatting with Ruff, type checking and automated tests in CI on every push/PR.

---

## Results

Evaluation on the **MEVID** benchmark (Multi-view Extended Videos with Identities, WACV 2023), official test split (54 identities, 1,754 tracklets, 316 queries):

| Metric | Value |
| :--- | ---: |
| Rank-1 | **52.53%** |
| Rank-5 | 66.77% |
| Rank-10 | 72.78% |
| Rank-20 | 80.70% |
| mAP | **27.01%** |

---

## Tech Stack

| Area | Technologies |
| :--- | :--- |
| Detection | YOLO26m-pose (Ultralytics) |
| Tracking | BoT-SORT |
| Re-Identification | C2DResNet50 + CAL (Clothes-Agnostic Learning), PyTorch / TorchVision |
| Clustering | Agglomerative hierarchical clustering (SciPy) |
| Acceleration | CUDA, TensorRT FP16 |
| Visualization | Streamlit, PyVis, Plotly, NetworkX, Pandas |
| Quality | Ruff, mypy, Pyright, pytest, GitHub Actions |

---

## Quick Start

**Requirements:** Python ≥ 3.10, NVIDIA GPU with CUDA ≥ 12.x (tested on RTX 3060 6 GB), Conda.

```bash
# 1. Clone the repository
git clone https://github.com/kiyx/ariadne-tracking.git
cd ariadne-tracking

# 2. Create the environment
conda create -n ariadne_env python=3.10 -y
conda activate ariadne_env

# 3. PyTorch with CUDA + dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 4. (Optional) Dashboard and development tools
pip install -e ".[viz,dev]"
```

**Run the full pipeline:**

```bash
python -m src.00_setup_dataset --subset   # download MEVID subset (~390 MB)
python -m src.01_tracker_extractor        # detection + tracking → ROIs
python -m src.02_feature_extractor        # Re-ID embeddings 2048-D
python -m src.03_build_graph              # cross-camera graph → output/global_graph.json
streamlit run src/04_visualize_graph.py   # dashboard → http://localhost:8501
```

Each module exposes CLI options (similarity thresholds, batch size, TensorRT, video limits, etc.) documented in `--help`. Default parameters are centralized in [`src/config.py`](src/config.py).

---

## Project Structure

```text
ariadne-tracking/
├── src/
│   ├── 00_setup_dataset.py      # MEVID video download from S3
│   ├── 01_tracker_extractor.py  # Detection + tracking → ROI extraction
│   ├── 02_feature_extractor.py  # C2DResNet50 + CAL → 2048-D embedding
│   ├── 03_build_graph.py        # 3-phase hierarchical clustering → JSON graph
│   ├── 04_visualize_graph.py    # Interactive dashboard (Streamlit)
│   ├── dashboard_viz.py         # Plotly/PyVis visualizations
│   ├── graph_queries.py         # Graph queries and statistics
│   ├── eval_mevid.py            # Official MEVID benchmark (CMC + mAP)
│   └── config.py                # Centralized constants and parameters
├── models/                      # YOLO and CAL weights + Simple-CCReID source
├── data/                        # Raw videos, extracted ROIs, MEVID annotations
├── output/                      # Graphs, reports and logs
├── tests/                       # pytest test suite
└── .github/workflows/           # CI: lint + type check + test
```

---

## Dataset

**MEVID** (Multi-view Extended Videos with Identities) is a multi-day, multi-camera surveillance dataset with **158 identities**, **598 outfits**, **8,092 tracklets** and **33 cameras**. The project includes automatic video download and an evaluation script that reproduces the official benchmark protocol.

---

## Development and Quality

```bash
ruff check .        # linting
ruff format .       # formatting
mypy src/           # type checking
pytest              # test suite
```

The CI (GitHub Actions) automatically runs linting, format check, type checking and tests on every push and pull request to the `main` branch.

---

## License

Thesis project — University of Naples Federico II.
