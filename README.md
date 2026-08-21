# GemmaBe — Brain Encoding con Gemma 4

---

## ¿Qué es esto?

Este proyecto implementa un pipeline de **brain encoding** que predice la actividad cerebral (fMRI) a partir de estímulos multimodales (video, audio, texto). La arquitectura se basa en:

- **Encoder congelado**: Gemma 4 E2B-it (5.1B parámetros) extrae embeddings de 1536 dimensiones por TR.
- **Bottleneck**: Reducción de dimensionalidad de 1536D → 512D.
- **Transformer Temporal**: Captura dependencias temporales en los TRs, con 8 cabezas de atención y 8 capas.
- **TailModel**: Entrenable, mapea 512D → 1000 parcelas de fMRI.
- **PyTorch Lightning**: Todo el entrenamiento es escalable, con logging automático y checkpoints.

El objetivo es demostrar la viabilidad de Gemma 4 para brain encoding en el dataset Algonauts 2025 (pilot subset).

> **Nota sobre evaluación**: Usamos un **hold-out estricto por episodios** (Season 6 de Friends como test, nunca visto durante entrenamiento), con alineación hemodinámica (HRF delay = 5s). Esto evita data leakage por ventanas temporales superpuestas y es comparable con la metodología de TriBE v1.

---

## Estructura del proyecto

```
GemmaBe/
├── src/                          # Código fuente
│   ├── models/                   # LightningModules
│   │   ├── base_module.py        # Lógica compartida (train/val/test steps)
│   │   ├── temporal_full_model.py         # Full con Transformer temporal
│   │   └── without_temporal_full_model.py # Full pointwise (sin Transformer)
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
│   ├── prepare_train_test_split.py  # Split estricto train/test (Season 6 hold-out)
│   └── config.py                 # Hiperparámetros centralizados
├── cloud/                        # Scripts para ejecución en RunPod
│   ├── setup.sh                  # Instala dependencias, verifica GPU
│   ├── run_extraction.sh         # Lanza extracción v2 con nohup
│   ├── list_pilot_files.py       # Genera lista de archivos a transferir
│   └── CLOUD_GUIDE.md            # Guía completa Mac → RunPod → Mac
├── train.py                      # Entrenamiento masivo (PyTorch Lightning)
├── evaluate_sequential.py        # Evaluación secuencial (últimos N TRs)
├── plots/                        # Generación de figuras
│   └── generate_figures.py       # Script para figuras del paper
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
- `data/features/` — features multimodales narrativos
- `data/features_text_only/` — baseline solo texto

### Paso 2: Descargar resultados a tu Mac

```bash
rsync -avz --progress root@RUNPOD_IP:/workspace/GemmaBe/data/features/ ./data/features/
rsync -avz --progress root@RUNPOD_IP:/workspace/GemmaBe/data/features_text_only/ ./data/features_text_only/
```

### Paso 3: Preparar split train/test estricto (Season 6 hold-out)

> **Importante**: Ya no usamos random split. La metodología anterior sufría de data leakage por ventanas superpuestas (stride=5 con window_size=67), produciendo scores inflados (~0.80 Pearson, por encima del techo de ruido biológico ~0.54).
>
> La nueva metodología usa **Season 6 como hold-out estricto**: episodios completos, nunca vistos durante entrenamiento.

```bash
# Prepara el split (automático para sub-01 y sub-02)
uv run python -m src.prepare_train_test_split
```

Esto genera:
- `data/train_test_split/features_train.pt` — Features de entrenamiento (S1-S5 + Movies, ~100k TRs)
- `data/train_test_split/features_test.pt` — Features de test (Season 6, ~23k TRs)
- `data/train_test_split/bold_train_sub-01.pt` — fMRI train sub-01 (normalizado con z-score fit en train)
- `data/train_test_split/bold_test_sub-01.pt` — fMRI test sub-01
- `data/train_test_split/bold_train_sub-02.pt` — fMRI train sub-02
- `data/train_test_split/bold_test_sub-02.pt` — fMRI test sub-02
- `data/train_test_split/split_info.json` — Metadatos del split

### Paso 4: Entrenar

```bash
# Grid completo para el paper (8 combinaciones)
uv run python train.py
```

---

## Entrenamiento

### Grid para el paper (8 combinaciones)

```bash
uv run python train.py --epochs 100 --batch_size 16 --stride 5
```

| Modelo | Estímulos | Sujetos | Runs | Qué mide |
|--------|-----------|---------|------|----------|
| `temporal_full` | multimodal, textonly | sub-01, sub-02 | 4 | **Modelo principal**: Bottleneck + Transformer temporal |
| `without_temporal_full` | multimodal, textonly | sub-01, sub-02 | 4 | **Efecto del Transformer**: pointwise vs temporal |
| **Total** | | | **8** | |

### Comparaciones clave del paper

| Comparación | Modelos | Pregunta |
|---|---|---|
| **Transformer vs Pointwise** | `temporal_full` vs `without_temporal_full` (mismo estímulo, mismo sujeto) | ¿Cuánto mejora el Transformer temporal? |
| **Multimodal vs Text-only** | Cualquier modelo: multimodal vs textonly (mismo sujeto) | ¿Qué aporta la información visual/auditiva? |

### Ablation: sin alineación hemodinámica (`no_hrf`)

Se entrenó un modelo `no_hrf` (misma arquitectura temporal pero sin HRF delay) como ablation exploratoria. Los resultados mostraron que el Transformer es robusto a desalineaciones moderadas, aprendiendo a compensar el desfase hemodinámico dentro de su ventana receptiva de ~100 segundos. Sin embargo, **todos los modelos principales usan HRF delay = 5s** por defecto.

### Comandos útiles

```bash
# Ver la grid sin entrenar
uv run python train.py --dry_run

# Solo algunos modelos
uv run python train.py --models temporal_full without_temporal_full

# Solo multimodal (omite textonly)
uv run python train.py --stimuli multimodal

# Ajustar hiperparámetros
uv run python train.py --epochs 200 --batch_size 128 --lr 5e-5
```

---

## Metodología: Alineación Hemodinámica (HRF)

Todos los modelos principales aplican **alineación hemodinámica** con un delay de **5 segundos** (≈3.4 TRs a 1.49s/TR), implementado en `src/utils/temporal_alignment.py`:

```python
from src.utils.temporal_alignment import HRFAligner

aligner = HRFAligner(hrf_peak_delay=5.0, tr=1.49)
# Alinea features al fMRI: los features del TR t se emparejan con fMRI del TR t + delay
```

Esto resuelve el desfase inherente entre el estímulo (percepción inmediata) y la respuesta BOLD (pico a ~5s). Los modelos `without_temporal` también usan este delay (pointwise, no hay ventana temporal).

### Por qué Season 6 hold-out

| Aspecto | Random 90/10 (anterior) | Season 6 hold-out (actual) |
|---|---|---|
| **Data leakage** | Sí — ventanas de 67 TRs con stride=5 se superponen entre train/val | No — episodios completos separados |
| **Scores Pearson** | ~0.80 (inflados, por encima del techo de ruido ~0.54) | ~0.15-0.35 (realistas, comparable a TriBE v1: 0.32) |
| **Validación** | No mide generalización a nuevos episodios | Sí — mide generalización a contenido nunca visto |
| **Tiempo de test** | ~3h (subset random) | ~9.5h (Season 6 completa) |

---

## Modelos

| Modelo | Descripción | Estímulos | HRF |
|--------|-------------|-----------|-----|
| `temporal_full` | Bottleneck(1536→512) + Transformer temporal(8 capas) + Head(512→1000) | both | 5s |
| `without_temporal_full` | Bottleneck(1536→512) + MLP pointwise(2 capas) + Head(512→1000) | both | 5s |

**Nota arquitectónica**: La ablation `without_temporal` mantiene una capacidad expresiva comparable pero sin mecanismo de atención temporal. Esto aísla el efecto del Transformer vs un MLP pointwise.

---

## Resultados guardados

Por cada experimento se genera:

```
results/
├── temporal_full_multimodal_sub-01/
│   ├── checkpoints/
│   │   └── temporal_full_multimodal_sub-01_epoch=042_pearson=0.2841.ckpt
│   ├── metrics/
│   │   ├── pearson_map_val.pt      # (1000,) correlación por parcela (val set)
│   │   ├── pearson_map_test.pt      # (1000,) correlación por parcela (Season 6 hold-out)
│   │   └── test_results.json       # métricas agregadas (test Pearson mean/std)
│   └── logs/
│       ├── tb/                     # TensorBoard
│       └── csv/                    # CSV por época
├── temporal_full_textonly_sub-01/
│   └── ...
```

### Evaluación en hold-out

```bash
# Evaluación secuencial (últimos N TRs, como sanity check)
uv run python evaluate_sequential.py --subject sub-01 --model_path results/temporal_full_multimodal_sub-01/checkpoints/best.ckpt
```

### Figuras para el paper

```bash
# Generar todas las figuras
uv run python plots/generate_figures.py

# Generar figura específica
uv run python plots/generate_figures.py --figure 2
uv run python plots/generate_figures.py --figure 3 --subject sub-01
uv run python plots/generate_figures.py --figure 4
```

**Figuras generadas** (`plots/`):
- `figure_2_bars.{png,pdf}` — Barras comparativas de Pearson por modelo y estímulo
- `figure_3_brain_maps_sub-01.{png,pdf}` — Mapas de superficie cortical (Schaefer-1000) para estímulo multimodal
- `figure_4_training_curves.{png,pdf}` — Curvas de entrenamiento (loss + Pearson)

### Visualizar logs

```bash
# TensorBoard
uv run tensorboard --logdir=results

# Leer CSV con pandas
import pandas as pd
df = pd.read_csv("results/.../logs/csv/temporal_full/version_0/metrics.csv")
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
- **Metodología de evaluación**: Basada en TriBE v1 (hold-out por episodios + HRF alignment)


# 1. Temporal models (4 experiments, ~90 min each with stride=5)


uv run python train.py \
  --models temporal_full \
  --stimuli multimodal textonly \
  --subjects sub-01 sub-02 \
  --epochs 110 \
  --batch_size 64 \
  --stride 5

# 2. Pointwise models (4 experiments, ~20 min each)

uv run python train.py \
  --models without_temporal_full \
  --stimuli multimodal textonly \
  --subjects sub-01 sub-02 \
  --epochs 110 \
  --batch_size 64
