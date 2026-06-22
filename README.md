# Ariadne Tracking

[![CI](https://github.com/kiyx/ariadne-tracking/actions/workflows/ci.yml/badge.svg)](https://github.com/kiyx/ariadne-tracking/actions/workflows/ci.yml)

## Multi-Camera Person Tracking & Re-Identification

> **Tesi di Laurea Triennale in Informatica**
> **Università degli Studi di Napoli Federico II**

|                     |                         |
| :------------------ | :---------------------- |
| **Candidato**       | Giuseppe Paolo Esposito |
| **Matricola**       | N86005174               |
| **Relatore**        | Prof. Daniel Riccio     |
| **Anno Accademico** | 2025/2026               |

---

## Panoramica

Pipeline modulare per il tracking multi-camera e la re-identificazione di
persone su video di sorveglianza. Il sistema elabora video grezzi dal dataset
**MEVID**, estrae tracklet per persona, calcola embedding Re-ID con un modello
*Clothes-Agnostic* (CAL) e costruisce un grafo di corrispondenza cross-camera.

```text
                    Modulo 0                    Modulo 1
               ┌──────────────┐    ┌───────────────────────────┐
 Video MEVID ──┤  S3 Download ├───►│ YOLO26m + BoT-SORT        │
               └──────────────┘    │ Detection → Tracking → ROI│
                                   └─────────────┬─────────────┘
                                                  │
                    Modulo 2                      ▼
               ┌──────────────────────────────────────┐
               │ C2DResNet50 + CAL → Embedding 2048-D │
               └──────────────┬───────────────────────┘
                              │
                    Modulo 3  ▼
               ┌──────────────────────────────────────┐
               │           Grafo JSON                 │
               └──────────────────────────────────────┘

```

## Requisiti

| Requisito        | Versione minima                        |
| :--------------- | :------------------------------------- |
| **Python**       | ≥ 3.10                                 |
| **NVIDIA GPU**   | CUDA ≥ 12.x (testato su RTX 3060 6 GB) |
| **Conda**        | Miniconda / Anaconda                   |
| **Spazio disco** | ~390MB (subset 3 video)                |

## Installazione

```bash
# 1. Clona il repository
git clone https://github.com/kiyx/ariadne-tracking.git
cd ariadne-tracking

# 2. Ambiente conda
conda create -n ariadne_env python=3.10 -y
conda activate ariadne_env

# 3. PyTorch con CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 4. Dipendenze progetto
pip install -r requirements.txt

# 5. (Opzionale) Dashboard interattiva (Streamlit)
pip install -e ".[viz]"

# 6. (Opzionale) Tool di sviluppo (pytest, ruff, mypy)
pip install -e ".[dev]"
```

## Quick Start

```bash
conda activate ariadne_env

# Download subset MEVID (3 video, ~390 MB)
python -m src.00_setup_dataset --subset

# Pipeline completa
python -m src.01_tracker_extractor          # Detection + tracking → ROI
python -m src.02_feature_extractor          # Embedding Re-ID 2048-D
python -m src.03_build_graph                # Grafo cross-camera
streamlit run src/04_visualize_graph.py     # Dashboard → http://localhost:8501
```

Per usare **TensorRT FP16** (accelera la detection ~2×):

```bash
python -m src.01_tracker_extractor --tensorrt
```

## Struttura

```text
ariadne-tracking/
├── src/
│   ├── config.py                  # Costanti e parametri centralizzati
│   ├── utils.py                   # Funzioni helper (bbox, logging, ROI)
│   ├── 00_setup_dataset.py        # Download video MEVID da S3
│   ├── 01_tracker_extractor.py    # YOLO26 + BoT-SORT → ROI extraction
│   ├── 02_feature_extractor.py    # C2DResNet50 + CAL → embedding 2048-D
│   ├── 03_build_graph.py          # Grafo di movimento cross/intra-camera
│   ├── 04_visualize_graph.py      # Dashboard interattiva (Streamlit)
│   └── eval_mevid.py              # Benchmark MEVID ufficiale (CMC + mAP)
├── models/
│   ├── yolo26m.pt                 # Pesi YOLO26m (detection)
│   ├── CAL_best_model.pth.tar     # Pesi CAL (Re-ID)
│   └── simple_ccreid/             # Sorgente Simple-CCReID (CAL backbone)
├── data/
│   ├── raw/videos/                # Video MEVID grezzi
│   ├── processed/extracted_rois/  # ROI estratte per tracklet
│   ├── mevid-v1-bbox-test/        # Crop GT per benchmark ufficiale
│   └── mevid-v1-annotation-data/  # Annotazioni MEVID
├── output/                        # Report JSON e log
├── pyproject.toml                 # Configurazione progetto e tool
└── requirements.txt               # Dipendenze pip
```

## Opzioni CLI

### `01_tracker_extractor` — Detection & Tracking

| Flag           | Default           | Descrizione                       |
| :------------- | :-----------      | :-------------------------------- |
| `--model`      | `yolo26m-pose.pt` | Modello YOLO                      |
| `--conf`       | `0.5`             | Soglia confidenza detection       |
| `--frame-skip` | `3`               | Salva 1 ROI ogni N frame          |
| `--imgsz`      | `640`             | Risoluzione input YOLO            |
| `--tensorrt`   | off               | Accelerazione TensorRT FP16       |
| `--no-resize`  | off               | Mantieni dimensione originale ROI |
| `--force`      | off               | Riprocessa video già completati   |
| `--show`       | off               | Preview detection in tempo reale  |
| `--max-videos` | `0` (tutti)       | Limita elaborazione a N video     |

### `02_feature_extractor` — Embedding Re-ID

| Flag           | Default                  | Descrizione                    |
| :------------- | :----------------------- | :----------------------------- |
| `--weights`    | `CAL_best_model.pth.tar` | Pesi modello CAL               |
| `--batch-size` | `8`                      | Batch size DataLoader          |
| `--workers`    | `2`                      | Worker DataLoader              |
| `--force`      | off                      | Rielabora video già completati |

### `03_build_graph` — Costruzione Grafo

| Flag             | Default | Descrizione                                   |
| :--------------- | :------ | :-------------------------------------------- |
| `--threshold`    | `0.55`  | Soglia similarità minima per clustering       |
| `--max-time-gap` | `300`   | Intervallo temporale massimo tra tracklet (s) |

### `04_visualize_graph` — Dashboard Streamlit

| Flag       | Default              | Descrizione                  |
| :--------- | :------------------- | :--------------------------- |
| `--graph`  | primo `graph_*.json` | Path specifico al grafo      |
| `--roi-dir`| `data/processed`     | Directory con ROI/thumbnails |

Avvio: `streamlit run src/04_visualize_graph.py -- --graph output/graph_*.json`

### `eval_mevid` — Benchmark Ufficiale

| Flag                   | Default | Descrizione                  |
| :--------------------- | :------ | :--------------------------- |
| `--clips-per-tracklet` | `4`     | Clip campionate per tracklet |
| `--batch-size`         | `16`    | Batch size DataLoader        |

## Parametri della Pipeline

Costanti configurabili in [`src/config.py`](src/config.py):

| Parametro                    | Valore  | Utilizzo                                     |
| :--------------------------- | :------ | :------------------------------------------- |
| `MIN_WIDTH` × `MIN_HEIGHT`   | 25×75   | Dimensione minima bbox accettata             |
| `ROI_RESIZE`                 | 128×256 | Dimensione output ROI (w × h)                |
| `PADDING_RATIO`              | 0.05    | Padding percentuale attorno alla bbox        |
| `SHARPNESS_THRESHOLD`        | 15      | Soglia Laplaciana per scartare ROI sfocate   |
| `CONTAINMENT_THRESHOLD`      | 0.5     | Soglia soppressione bbox contenute           |
| `MIN_TRACK_FRAMES`           | 8       | Frame minimi per considerare un track valido |
| `MIN_INTRA_TRACK_SIMILARITY` | 0.4     | Similarità coseno minima intra-track         |
| `SEQ_LEN`                    | 8       | Lunghezza sequenza temporale per C2DResNet   |
| `DEFAULT_SAMPLING_STRIDE`    | 4       | Stride temporale sampling clip               |
| `JPEG_QUALITY`               | 95      | Qualità di compressione JPEG                 |
| `DEFAULT_SEED`               | 67      | Seed per riproducibilità                     |

## Tecnologie

| Componente        | Tecnologia                                    |
| :---------------- | :-------------------------------------------- |
| Detection         | YOLO26m-pose (Ultralytics)                    |
| Tracking          | BoT-SORT                                      |
| Re-Identification | C2DResNet50 + CAL (Clothes-Agnostic Learning) |
| Clustering        | Agglomerativo (scipy)                         |
| Framework DL      | PyTorch + TorchVision                         |
| Accelerazione     | CUDA, TensorRT FP16                           |
| Dashboard         | Streamlit + PyVis + Plotly                    |

## Dataset

**MEVID** (Multi-view Extended Videos with Identities) — WACV 2023

- 158 identità, 598 outfit, 8 092 tracklet, 33 telecamere
- Test split: 54 identità, 1 754 tracklet, 316 query

### Risultati Benchmark MEVID (protocollo standard)

| Metrica | Valore  |
| :------ | -----:  |
| Rank-1  |  52.53% |
| Rank-5  |  66.77% |
| Rank-10 |  72.78% |
| Rank-20 |  80.70% |
| mAP     |  27.01% |

## Licenza

Progetto di tesi — Università degli Studi di Napoli Federico II.
