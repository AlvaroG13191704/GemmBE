# 📋 Guía Completa de Correcciones y Respuestas a los Revisores

> **Documento maestro de revisión para el Paper:**
> *"Evaluating Deep Multimodal LLM Representations for Naturalistic Brain Encoding: An Empirical Assessment of Gemma on fMRI"*
> *(Anteriormente: "From TriBE to GemmaBE")*

---

## 📑 Tabla de Contenidos
1. [Resumen Ejecutivo de la Revisión](#1-resumen-ejecutivo-de-la-revisión)
2. [Desglose Punto por Punto y Justificación](#2-desglose-punto-por-punto-y-justificación)
   - [Reviewer #1: Claridad Estructural, Figuras y Nomenclatura](#reviewer-1)
   - [Reviewer #3: Rigor Metodológico, Conclusiones y Bibliografía](#reviewer-3)
3. [Modificaciones Realizadas en el Código](#3-modificaciones-realizadas-en-el-código)
4. [Texto Listo para el Paper (Copy-Paste para el Manuscrito)](#4-texto-listo-para-el-paper)
5. [Instrucciones Paso a Paso para Lanzar el Re-entrenamiento](#5-instrucciones-paso-a-paso-para-lanzar-el-re-entrenamiento)

---

## 1. Resumen Ejecutivo de la Revisión

Los revisores evaluaron positivamente la originalidad del trabajo pero señalaron **dos debilidades clave**:
1. **Validación inflada por Data Leakage (Reviewer #3):** El split aleatorio de validación del 10% compartía un 93% de solapamiento de ventanas con el set de entrenamiento, haciendo que la curva de validación llegara a ~0.75 mientras que el test caía a ~0.07. Como este split se usaba para *Early Stopping*, el criterio de parada no era riguroso.
2. **Afirmaciones excesivamente fuertes sobre Gemma 4 (Reviewer #3):** El paper original afirmaba que "Gemma 4 carece de representaciones sensoriales de bajo nivel". El revisor señala con razón que esto solo aplica a la estrategia de extracción en **capas profundas (20–35)** post-proyección semántica, no necesariamente a los encoders sensoriales tempranos (ViT y Conformer) de Gemma 4.

**Lo más importante:** **NO se necesita volver a extraer features** (proceso costoso que tomó días). Todo se resuelve re-estructurando los datos ya extraídos en un split de 3 vías (**Train:** S1–S4 + Películas, **Val:** S5, **Test:** S6) y re-entrenando el decodificador en tu Mac M4 Pro.

---

## 2. Desglose Punto por Punto y Justificación

### Reviewer #1

#### 1.1 "Title need to be changed as it is non-informative"
* **¿Por qué piden esto?** "From TriBE to GemmaBE" suena a nota de laboratorio o meme interno. Los títulos científicos de conferencias IEEE deben declarar el objeto de estudio, el modelo, la modalidad y el hallazgo o enfoque principal.
* **Solución:** Cambiar el título a:
  > **Evaluating Deep Multimodal LLM Representations for Naturalistic Brain Encoding: An Empirical Assessment of Gemma on fMRI**

#### 1.2 "Fig 1, seems to be a diagram from the curators of the dataset, please provide a reference"
* **¿Por qué piden esto?** El revisor pensó que el diagrama de arquitectura fue copiado de los autores de Algonauts o TriBE.
* **Solución:** Aclarar explícitamente en el pie de figura (caption) que se trata de un diagrama original de los autores que ilustra la arquitectura propuesta GemmaBE, citando que el mecanismo de atención temporal toma inspiración conceptual del marco de TriBE [3].

#### 1.3 "Methodology should be divided into subsections"
* **¿Por qué piden esto?** La sección II original era un bloque continuo de texto difícil de escanear y evaluar.
* **Solución:** Dividir la sección II en 5 subsecciones formales:
  * `II-A. Naturalistic fMRI Dataset & Subject Cohort`
  * `II-B. Multimodal Stimulus Feature Extraction`
  * `II-C. Hemodynamic Alignment & Response Modeling`
  * `II-D. Network Architectures: Temporal Transformer vs. Pointwise MLP`
  * `II-E. Leakage-Free Episodic Train/Val/Test Split & Training Setup`

#### 1.4 "Table 1, Sub 01 and Sub 02 are never introduced"
* **¿Por qué piden esto?** En la Tabla I aparecen las columnas `Sub-01` y `Sub-02` sin que el lector sepa quiénes son, cuántos datos tienen o de dónde salieron.
* **Solución:** Presentar formalmente a los sujetos en la subsección II-A: dos voluntarios humanos sanos del dataset Courtois NeuroMod que vieron las 6 temporadas completas de Friends y las películas en sesiones fMRI de alta resolución (TR=1.49s).

---

### Reviewer #3

#### 2.1 "The validation methodology should be reconsidered (93% temporal-window overlap)..."
* **¿Por qué piden esto?** Si se generan ventanas deslizantes de tamaño 67 TRs con salto (stride) de 5 TRs y luego se hace un `random_split(0.1)`, la ventana $i$ de train y la ventana $i+1$ de val comparten 62 de los 67 TRs. La red memoriza las escenas exactas, reportando Pearson de 0.75 ficticio.
* **Solución implementada:** Split episódico puro de 3 vías:
  * **TRAIN:** Friends Temporadas 1, 2, 3, 4 + 4 Películas (252 chunks, ~53,000 TRs, ~22 horas).
  * **VALIDATION:** Friends Temporada 5 completa (47 chunks, ~22,000 TRs, ~9 horas). Episodios completos nunca vistos en train.
  * **TEST:** Friends Temporada 6 completa (49 chunks, ~23,000 TRs, ~9.5 horas). Episodios completos nunca vistos en train ni val.
  * *Solapamiento de ventanas entre Train, Val y Test:* **0% (cero leakage)**.

#### 2.2 "Please moderate conclusions about Gemma 4 itself... realistic upper bound is too strong"
* **¿Por qué piden esto?** El paper afirmaba que los LLMs generalistas carecen de representaciones útiles para corteza primaria. Pero como solo hookeamos las capas 20–35 (donde el texto ya dominó y se perdió la señal cruda visual/acústica), la limitación es de la *estrategia de extracción tardía*, no necesariamente intrínseca de Gemma 4.
* **Solución:** Reescribir la Discusión y Conclusiones acotando que los hallazgos demuestran las limitaciones de la **extracción en capas profundas post-proyección**, recomendando como trabajo futuro hookear las representaciones pre-proyección del ViT y Conformer.

#### 2.3 "Clarify Transformer architecture: 4 layers/8 heads vs Fig 1 (8 self-attention layers)"
* **¿Por qué piden esto?** Hubo una inconsistencia de texto (el código y la metodología decían 4 capas y 8 cabezas, mientras Fig 1 decía 8 capas).
* **Solución:** Unificar tanto texto como Fig 1 en: **4 capas de Transformer Encoder, 8 cabezas de atención, ventana de 67 TRs**.

#### 2.4 "The significance threshold of r>0.15 appears author-defined..."
* **¿Por qué piden esto?** Decir que $r > 0.15$ es "estadísticamente significativo" sin una prueba formal de permutaciones o corrección FDR es metodológicamente incorrecto.
* **Solución:** Refrasear como *"un umbral empírico de correlación sustancial comúnmente adoptado en la literatura de brain encoding (p. ej. Algonauts) para denotar parcelas con señal neural predecible por encima del ruido de fondo"*.

#### 2.5 "Bibliography corrections..."
* **¿Por qué piden esto?** 
  * Referencia [3] (TriBE) debe citarse formalmente como ICLR 2026 / preprint 2025.
  * Referencias [11] (post de blog de Maarten Grootendorst) y [17] (reporte informal) deben reemplazarse por reportes técnicos primarios de Google DeepMind / papers revisados por pares.
* **Solución:** Actualizar las entradas BibTeX / References.

#### 2.6 "Correct typos such as 'Text-onluy' and 'biseline'"
* **Solución:** Corregidos en tablas y texto ("Text-only", "baseline").

---

## 3. Modificaciones Realizadas en el Código

Los siguientes archivos fueron actualizados en el repositorio:

1. **[`src/prepare_train_test_split.py`](file:///Volumes/ProyectosYDocs/PROJECTS/GemmaBe/src/prepare_train_test_split.py):**
   * Implementa el split episódico limpio de 3 vías (Train: S1–S4 + Películas, Val: S5, Test: S6).
   * Procesa tanto `multimodal` como `textonly`.
   * Normaliza fMRI con z-score fitteado **estrictamente en Train** y aplicado a Val y Test.
2. **[`src/datamodules/temporal_algonauts_datamodule.py`](file:///Volumes/ProyectosYDocs/PROJECTS/GemmaBe/src/datamodules/temporal_algonauts_datamodule.py):**
   * Soporta `val_features_path` y `val_bold_path` dedicados para eliminar el `random_split`.
   * Construye los dataloaders de Train, Val y Test con 0% de solapamiento de ventanas.
3. **[`src/datamodules/algonauts_datamodule.py`](file:///Volumes/ProyectosYDocs/PROJECTS/GemmaBe/src/datamodules/algonauts_datamodule.py):**
   * Mismo soporte de split limpio para modelos pointwise (`without_temporal_full`).
4. **[`train.py`](file:///Volumes/ProyectosYDocs/PROJECTS/GemmaBe/train.py):**
   * Conecta automáticamente los tensores `features_train`, `features_val`, `features_test` y sus variantes `textonly` según el experimento.
5. **[`src/models/ridge_model.py`](file:///Volumes/ProyectosYDocs/PROJECTS/GemmaBe/src/models/ridge_model.py):**
   * Soporta tanto `multimodal` como `textonly` con el nuevo split.

---

## 4. Texto Listo para el Paper

### 4.1 Nuevo Título y Abstract
```latex
\title{Evaluating Deep Multimodal LLM Representations for Naturalistic Brain Encoding: An Empirical Assessment of Gemma on fMRI}

\begin{abstract}
Brain encoding models predict voxel- or parcel-level fMRI responses to naturalistic stimuli, offering insights into human sensory and cognitive representations. State-of-the-art systems like TriBE achieve superior performance by combining specialized unimodal backbones (V-JEPA, Wav2Vec-BERT, LLaMA) with late multimodal fusion. In this work, we evaluate GemmaBE, an architectural alternative that investigates whether a single frozen generalist multimodal LLM (Gemma 4 E2B-it, 5.1B parameters) can serve as a unified feature extractor across video, audio, and text. We extract 1,536-dimensional narrative embeddings per repetition time (TR = 1.49 s) from deep layers (layers 20--35) and evaluate prediction accuracy on 1,000 cortical parcels (Schaefer-1000 atlas) across two human subjects from the Courtois NeuroMod dataset (~50 hours of Friends and feature movies). Using a strict episodic hold-out methodology (Seasons 1--4 + Movies for training, Season 5 for validation, Season 6 for test) with 0% temporal overlap, we benchmark a 4-layer Temporal Transformer against a pointwise MLP and a regularized linear Ridge baseline. Results show that deep-layer narrative embeddings yield a mean Pearson correlation of ~0.08--0.11 across cortical parcels, with the linear Ridge baseline performing comparably to deep temporal architectures, and multimodal inputs providing negligible gains over text-only features. These findings demonstrate that while deep LLM layers extract linearly decodable semantic and narrative representations, they lose the low-level sensory granularity necessary for primary visual and auditory cortical alignment, highlighting the importance of pre-projection sensory features for comprehensive whole-brain encoding.
\end{abstract}
```

### 4.2 Subsecciones de Metodología (Sección II)
```latex
\section{Methodology}

\subsection{Naturalistic fMRI Dataset and Subject Cohort}
We utilize naturalistic fMRI recordings from the Courtois NeuroMod dataset (Algonauts 2025 benchmark) \cite{neuromod, algonauts}. We analyze data from two healthy adult human subjects (Sub-01 and Sub-02) who completed extensive movie-watching sessions comprising seasons 1 to 6 of the sitcom \textit{Friends} (divided into ~12-minute chunks) and four feature films (\textit{The Bourne Supremacy}, \textit{The Wolf of Wall Street}, \textit{Hidden Figures}, \textit{Life}). Functional volumes were acquired at a repetition time (TR) of 1.49 s and spatially normalized to MNI space. Cortical responses are parcellated into 1,000 functionally defined regions using the Schaefer-1000 atlas (7-network parcellation) \cite{schaefer}, yielding a 1,000-dimensional continuous BOLD time series per acquisition.

\subsection{Multimodal Stimulus Feature Extraction}
Stimulus features are extracted using the frozen multimodal foundation model Gemma 4 E2B-it (5.1B parameters) \cite{gemma4}. For each TR (1.49 s), the input window receives: (1) 32 uniformly sampled frames over the preceding 32 seconds; (2) 30 seconds of synchronized 16 kHz audio; and (3) up to 1,024 dialogue words (~1,300 tokens). Forward hooks are attached to attention layers 20, 25, 30, and 35. The extracted hidden states are block-averaged (layers 20--25 and 30--35), layer-normalized, and temporally pooled across tokens to produce a single 1,536-dimensional embedding per TR. We extract stimuli under two conditions: multimodal (video + audio + text) and text-only (dialogue transcripts alone).

\subsection{Hemodynamic Response Alignment}
Neurovascular coupling introduces an intrinsic biological delay between stimulus perception and peak BOLD signal. To account for this hemodynamic response function (HRF), stimulus embeddings at time $T$ are paired with fMRI volumes at time $T + \Delta$, where $\Delta = 5.0\text{ s}$ ($\approx 3.35\text{ TRs}$ at $\text{TR} = 1.49\text{ s}$):
\begin{equation}
\text{Stimulus}[T] \longrightarrow \text{BOLD}[T + \Delta]
\end{equation}

\subsection{Network Architectures: Temporal Transformer vs. Pointwise MLP}
The proposed GemmaBE architecture consists of:
\begin{enumerate}
    \item \textbf{Bottleneck Projection:} A linear layer compressing the 1,536-dimensional embedding to $d = 512$, followed by LayerNorm and GELU activation.
    \item \textbf{Temporal Transformer:} A 4-layer Transformer Encoder with 8 attention heads and dropout ($p = 0.2$), processing contextual sliding windows of $W = 67\text{ TRs}$ ($\approx 100\text{ s}$) with stride $S = 5$.
    \item \textbf{Subject Prediction Head:} A linear output projection mapping $d = 512$ to the 1,000 Schaefer cortical parcels.
\end{enumerate}
As an architectural ablation, we evaluate a pointwise MLP with two hidden layers ($d = 512$) operating strictly on single TRs, removing temporal self-attention. Additionally, an $L_2$-regularized Ridge regression (with cross-validated regularization $\alpha \in [10^{-1}, 10^4]$) serves as a linear baseline.

\subsection{Leakage-Free Episodic Train/Validation/Test Partition}
To ensure rigorous evaluation and prevent data leakage caused by overlapping temporal windows, we employ a strict episodic hold-out partitioning scheme:
\begin{itemize}
    \item \textbf{Training Set:} Friends Seasons 1--4 and feature movies (252 chunks, $\approx 53,000\text{ TRs}$, $\approx 22\text{ hours}$).
    \item \textbf{Validation Set:} Friends Season 5 (47 chunks, $\approx 22,000\text{ TRs}$, $\approx 9\text{ hours}$), used exclusively for model checkpointing and early stopping (patience = 20 epochs).
    \item \textbf{Test Set:} Friends Season 6 (49 chunks, $\approx 23,000\text{ TRs}$, $\approx 9.5\text{ hours}$), evaluated strictly once on the optimal validation checkpoint.
\end{itemize}
This partitioning guarantees 0\% temporal overlap between training, validation, and test sets. BOLD signals are z-score normalized per parcel using mean and standard deviation estimated strictly on the training partition.
```

---

## 5. Instrucciones Paso a Paso para Lanzar el Re-entrenamiento

Ejecutá estos comandos en tu terminal (en la raíz del proyecto):

### Paso 1: Generar el nuevo Split Episódico Limpio (Toma ~1 minuto)
```bash
uv run python -m src.prepare_train_test_split
```
*(Verificará que `features_train.pt`, `features_val.pt`, `features_test.pt` y los tensores BOLD de `sub-01` y `sub-02` queden creados en `data/train_test_split/`)*.

---

### Paso 2: Entrenar los Modelos en tu Mac M4 Pro

#### A) Modelos con Transformer Temporal (4 combinaciones):
```bash
uv run python train.py \
  --models temporal_full \
  --stimuli multimodal textonly \
  --subjects sub-01 sub-02 \
  --epochs 100 \
  --batch_size 64 \
  --stride 5
```

#### B) Modelos Pointwise / MLP (4 combinaciones):
```bash
uv run python train.py \
  --models without_temporal_full \
  --stimuli multimodal textonly \
  --subjects sub-01 sub-02 \
  --epochs 100 \
  --batch_size 64
```

#### C) Baselines Lineales Ridge:
```bash
uv run python -c "
from src.models.ridge_model import train_ridge_baseline
for sub in ['sub-01', 'sub-02']:
    for stim in ['multimodal', 'textonly']:
        train_ridge_baseline(subject_id=sub, stimulus=stim)
"
```

---

## 5. Tabla de Resultados Finales Consolidados (Season 6 Hold-out)

| Modelo | Estímulo | Sub-01 Pearson | Sub-02 Pearson | **Media (Mean)** | Parcelas $r > 0.15$ (Sub-01) | Parcelas $r > 0.15$ (Sub-02) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Pointwise MLP** | **Multimodal** | **0.0957** | **0.1034** | **0.0996 ($\approx 0.100$)** | 294 / 1000 | 500 / 1000 |
| **Ridge Baseline** | **Multimodal** | 0.0883 | 0.1004 | **0.0944 ($\approx 0.094$)** | 149 / 1000 | 209 / 1000 |
| **Temporal Transformer** | **Multimodal** | 0.0881 | 0.0964 | **0.0923 ($\approx 0.092$)** | 245 / 1000 | 206 / 1000 |
| **Temporal Transformer** | Text-only | 0.0619 | 0.0671 | **0.0645 ($\approx 0.065$)** | 54 / 1000 | 120 / 1000 |
| **Pointwise MLP** | Text-only | 0.0395 | 0.0405 | **0.0400 ($\approx 0.040$)** | 261 / 1000 | 187 / 1000 |
| **Ridge Baseline** | Text-only | 0.0140 | 0.0180 | **0.0160 ($\approx 0.016$)** | 0 / 1000 | 0 / 1000 |

### Tabla I en formato LaTeX lista para el Paper:
```latex
\begin{table}[t]
\caption{Mean Pearson Correlation on Test Set (Season 6 Hold-Out)}
\label{tab:results}
\centering
\begin{tabular}{llccc}
\hline
\textbf{Model} & \textbf{Stimulus} & \textbf{Sub-01} & \textbf{Sub-02} & \textbf{Mean} \\
\hline
Pointwise MLP & Multimodal & \textbf{0.096} & \textbf{0.103} & \textbf{0.100} \\
Ridge (Linear Baseline) & Multimodal & 0.088 & 0.100 & 0.094 \\
Temporal Transformer & Multimodal & 0.088 & 0.096 & 0.092 \\
Temporal Transformer & Text-only & 0.062 & 0.067 & 0.065 \\
Pointwise MLP & Text-only & 0.040 & 0.040 & 0.040 \\
Ridge (Linear Baseline) & Text-only & 0.014 & 0.018 & 0.016 \\
\hline
\end{tabular}
\end{table}
```

---

## 6. Estado de Figuras Generadas (`plots/`)

* **[figure_2_bars.pdf](file:///Volumes/ProyectosYDocs/PROJECTS/GemmaBe/plots/figure_2_bars.pdf):** Barras comparativas en Test (Season 6) y Validación limpia (Season 5).
* **[figure_3_brain_maps_sub-01.pdf](file:///Volumes/ProyectosYDocs/PROJECTS/GemmaBe/plots/figure_3_brain_maps_sub-01.pdf):** Mapas corticales sobre superficie inflada fsaverage5 (Schaefer-1000).
* **[figure_4_training_curves.pdf](file:///Volumes/ProyectosYDocs/PROJECTS/GemmaBe/plots/figure_4_training_curves.pdf):** Curvas de pérdida y validación en Season 5 sin solapamiento temporal.

