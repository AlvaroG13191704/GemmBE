# GemmaBe — Brain Encoding con Gemma 4

> **MicroTRIBE-Gemma**: Simplificación de TriBE v2 usando Gemma 4 E2B-it como encoder multimodal congelado para Algonauts 2025.

---

## ¿Qué es esto?

Este proyecto implementa un pipeline de **brain encoding** que predice la actividad cerebral (fMRI) a partir de estímulos multimodales (video, audio, texto). La arquitectura se basa en:

- **Encoder congelado**: Gemma 4 E2B-it (5.1B parámetros) extrae embeddings de 1536 dimensiones por TR.
- **TailModel ligero**: Entrenable, mapea 1536D → 1000 parcelas de fMRI.
- **PyTorch Lightning**: Todo el entrenamiento es escalable, con logging automático y checkpoints.

El objetivo es demostrar la viabilidad de Gemma 4 para brain encoding en el dataset Algonauts 2025 (pilot subset).

---

## Estructura del proyecto

```
GemmaBe/
├── src/                          # Código fuente
│   ├── models/                   # LightningModules
│   │   ├── base_module.py        # Lógica compartida (train/val/test steps)
│   │   ├── temporal_full_model.py         # Full con Transformer temporal
│   │   ├── without_temporal_full_model.py # Full pointwise (sin Transformer)
│   │   └── ridge_model.py                 # Ridge Regression baseline
│   ├── datamodules/              # LightningDataModules
│   │   ├── algonauts_datamodule.py        # Datos pointwise (TR independiente)
│   │   └── temporal_algonauts_datamodule.py # Ventanas de TRs
│   ├── callbacks/                # Callbacks custom
│   │   └── metrics_callback.py   # Guarda Pearson maps + test_results.json
│   ├── architecture/             # Componentes arquitectónicos
│   │   └── temporal_transformer.py        # Transformer Encoder temporal
│   ├── utils/                    # Utilidades y procesamiento
│   │   ├── prepare_fmri.py       # Carga HDF5 de Algonauts
│   │   ├── temporal_alignment.py # HRFAligner + TemporalPooling
│   │   ├── helpers.py            # extract_audio, etc.
│   │   └── validate_tensors.py   # Validación de tensores
│   ├── extract_features_v2.py    # Extracción v2 (codificador narrativo)
│   ├── filter_fmri.py            # Filtra fMRI para chunks procesados
│   └── config.py                 # Hiperparámetros centralizados
├── cloud/                        # Scripts para ejecución en RunPod
│   ├── setup.sh                  # Instala dependencias, verifica GPU
│   ├── run_extraction.sh         # Lanza extracción v2 con nohup
│   ├── list_pilot_files.py       # Genera lista de archivos a transferir
│   └── CLOUD_GUIDE.md            # Guía completa Mac → RunPod → Mac
├── train.py                      # Entrenamiento masivo (PyTorch Lightning)
├── pyproject.toml                # Dependencias (incluye lightning, tensorboard)
├── README.md                     # Este archivo
└── AGENTS.md                     # Convenciones del proyecto
```

---

## Flujo completo: desde extracción hasta entrenamiento

### Paso 1: Extraer features con Gemma 4 (en RunPod)

```bash
# En tu Mac: generar lista de archivos
python cloud/list_pilot_files.py

# Transferir código + estímulos a RunPod
rsync -avz --progress --exclude='...' ./ root@RUNPOD_IP:/workspace/GemmaBe/
rsync -avz --progress --files-from=pilot_files.txt ./algonauts_2025/ root@RUNPOD_IP:/workspace/GemmaBe/algonauts_2025/

# En RunPod:
cd /workspace/GemmaBe
bash cloud/setup.sh
nohup bash cloud/run_extraction.sh > extraction_v2.log 2>&1 &
```

Esto genera:
- `data/features_v2/` — features multimodales narrativos (33 chunks)
- `data/features_v2_text_only/` — baseline solo texto (33 chunks)

### Paso 2: Descargar resultados a tu Mac

```bash
rsync -avz --progress root@RUNPOD_IP:/workspace/GemmaBe/data/features_v2/ ./data/features_v2/
rsync -avz --progress root@RUNPOD_IP:/workspace/GemmaBe/data/features_v2_text_only/ ./data/features_v2_text_only/
```

### Paso 3: Filtrar fMRI para sincronizar con chunks procesados

```bash
# Filtra sub-01 y sub-02 por defecto
python src/filter_fmri.py

# O especificar sujetos explícitamente
python src/filter_fmri.py --subjects sub-01 sub-02
```

Esto genera:
- `data/subjects_fmri_filtered/sub-01.pt`
- `data/subjects_fmri_filtered/sub-02.pt`
- Solo incluye los TRs de los 33 chunks extraídos.

### Paso 4: Entrenar

```bash
python train.py
```

---

## Entrenamiento

### Grid para el paper (14 combinaciones)

```bash
python train.py
```

| Modelo | Estímulos | Sujetos | Runs | Qué mide |
|--------|-----------|---------|------|----------|
| `temporal_full` | multimodal, textonly | sub-01, sub-02 | 4 | **Modelo principal**: Bottleneck + Transformer temporal |
| `without_temporal_full` | multimodal, textonly | sub-01, sub-02 | 4 | **Efecto del Transformer**: pointwise vs temporal |
| `no_hrf` | **multimodal** | sub-01, sub-02 | 2 | **Importancia del delay HRF**: sin alineación hemodinámica |
| `ridge` | multimodal, textonly | sub-01, sub-02 | 4 | **Baseline lineal**: Ridge Regression (sklearn) |
| **Total** | | | **14** | |

### Comparaciones clave del paper

| Comparación | Modelos | Pregunta |
|---|---|---|
| **Transformer vs Pointwise** | `temporal_full` vs `without_temporal_full` (mismo estímulo, mismo sujeto) | ¿Cuánto mejora el Transformer temporal? |
| **HRF alignment** | `without_temporal_full` vs `no_hrf` (ambos multimodal) | ¿Qué aporta alinear con el delay hemodinámico? |
| **Multimodal vs Text-only** | Cualquier modelo: multimodal vs textonly (mismo sujeto) | ¿Qué aporta la información visual/auditiva? |
| **Deep vs Linear** | `without_temporal_full` vs `ridge` (mismo estímulo) | ¿Cuánto gana la red profunda sobre Ridge? |

### Comandos útiles

```bash
# Ver la grid sin entrenar
python train.py --dry_run

# Solo algunos modelos
python train.py --models temporal_full without_temporal_full

# Solo multimodal (omite textonly)
python train.py --stimuli multimodal

# Ajustar hiperparámetros
python train.py --epochs 200 --batch_size 128 --lr 5e-5
```

---

## Modelos

| Modelo | Descripción | Estímulos |
|--------|-------------|-----------|
| `temporal_full` | Bottleneck(1536→512) + Transformer temporal(8 capas) + SubjectBlock | both |
| `without_temporal_full` | Bottleneck(1536→512) + SubjectBlock (sin Transformer) | both |
| `no_hrf` | Sin delay hemodinámico (HRF=0s). Mismo arquitectura que without_temporal. | multimodal only |
| `ridge` | Ridge Regression (sklearn) como baseline lineal | both |

---

## Resultados guardados

Por cada experimento se genera:

```
results/
├── temporal_full_multimodal_sub-01/
│   ├── checkpoints/
│   │   └── temporal_full_multimodal_sub-01_epoch=042_pearson=0.2841.ckpt
│   ├── metrics/
│   │   ├── pearson_map_val.pt      # (1000,) correlación por parcela
│   │   ├── pearson_map_test.pt
│   │   └── test_results.json       # métricas agregadas
│   └── logs/
│       ├── tb/                     # TensorBoard
│       └── csv/                    # CSV por época
├── temporal_full_textonly_sub-01/
│   └── ...
```

### Visualizar logs

```bash
# TensorBoard
tensorboard --logdir=results

# Leer CSV con pandas
import pandas as pd
df = pd.read_csv("results/.../logs/csv/metrics.csv")
```

---

## Dependencias

Instalación:
```bash
pip install -e .
```

Principales:
- `torch>=2.3.0`
- `lightning>=2.4.0`
- `transformers>=4.51.0`
- `scikit-learn>=1.5.0`
- `tensorboard>=2.17.0`

---

## Créditos

- **Dataset**: Algonauts 2025 Challenge
- **Modelo base**: Gemma 4 E2B-it (Google)
- **Arquitectura inspirada en**: TriBE v2 (Meta FAIR)
