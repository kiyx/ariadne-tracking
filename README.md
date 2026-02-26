# Ariadne Tracking System

## Distributed Multi-Camera Tracking & Re-Identification Framework

> **Tesi di Laurea Triennale in Informatica**
> **Università degli Studi di Napoli Federico II**

| | |
| :--- | :--- |
| **Candidato** | Giuseppe Paolo Esposito |
| **Matricola** | N86005174 |
| **Relatore** | Prof. Daniel Riccio |
| **Anno Accademico** | 2025/2026 |

---

## Descrizione del Progetto

## Architettura

## Requisiti

- **Python** ≥ 3.10
- **NVIDIA GPU** con CUDA ≥ 12.x (testato su RTX 3060 6GB)
- **Conda** Miniconda

## Installazione

```bash
# 1. Clona il repository
git clone https://github.com/kiyx/ariadne-tracking.git
cd ariadne-tracking

# 2. Crea l'ambiente conda
conda create -n ariadne_env python=3.10 -y
conda activate ariadne_env

# 3. Installa PyTorch con supporto CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 4. Installa le dipendenze del progetto
pip install -r requirements.txt

# 5. (Opzionale) Installa le dipendenze di sviluppo (pytest)
pip install -e ".[dev]"
```

## Struttura del Progetto

## Uso

### Esecuzione base

```bash
cd ariadne-tracking
python src/01_tracker_extractor.py
```

Usa i percorsi di default definiti in `configs/default.yaml`.

### Con TensorRT (consigliato per produzione)

```bash
# Prima esecuzione: esporta il modello (~2-5 min, poi riusa il .engine)
python src/01_tracker_extractor.py --tensorrt
```

### Opzioni CLI

## Tecnologie

## Dataset

## Licenza

Progetto di tesi — Università degli Studi di Napoli Federico II.
