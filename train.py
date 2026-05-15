"""
================================================================================
train.py — Entrenamiento offline del TailModel para Brain Encoding
================================================================================

PROPÓSITO GENERAL
-----------------
Entrena modelos de "cola" (head) que mapean features pre-extraídos de Gemma 4
(1536 dims) → actividad BOLD cerebral (1000 parcelas de Algonauts 2025).

CRÍTICO: Este script NO carga Gemma 4 en memoria. Los features ya fueron
extraídos por src/extract_features.py y guardados en disco como tensores .pt.
Esto hace el entrenamiento ~1000× más rápido: cientos de épocas en minutos.

MÓDULOS DE ENTRENAMIENTO
------------------------
El script soporta 4 modos de entrenamiento (seleccionables vía --mode):

  1. full (default)        → TailModel completo: Bottleneck(1536→512) + TemporalTransformer(512→512) + SubjectBlock(512→1000)
  2. ridge                 → Ridge Regression (sklearn) como baseline lineal
  3. no_bottleneck         → Linear(1536→1000) directo, sin compresión intermedia
  4. no_hrf                → TailModel completo pero SIN desfase hemodinámico (shift=0)

MÉTRICA PRINCIPAL
-----------------
Pearson correlation per parcel (per-vertex): mide qué tan bien la FORMA temporal
de la predicción coincide con la actividad BOLD real. Es invariante a escala
y offset, lo cual es crucial porque la señal BOLD no tiene unidades absolutas.

FLUJO DE EJECUCIÓN
------------------
1. Carga features y BOLD desde disco.
2. Aplica HRFAligner (shift temporal de ~5s / 3 TRs).
3. Separa train/val (80/20 por defecto).
4. Entrena el modelo seleccionado.
5. Calcula Pearson por parcela y guarda mapa en disco.
6. Reporta métricas y guarda curvas de entrenamiento.

OUTPUTS GUARDADOS
-----------------
En cada ejecución se crea un subdirectorio en `results/`:
  results/full_sub-01_20240505_143022/
    ├── model.pt              → Pesos del modelo entrenado
    ├── pearson_map.pt        → Vector de Pearson por parcela (1000,)
    ├── history.json          → Curvas de loss y Pearson por época
    └── config.json           → Hiperparámetros usados

USO
---
    # TailModel completo, 1 sujeto
    uv run python train.py --subject sub-01 --mode full

    # Ridge regression baseline
    uv run python train.py --subject sub-01 --mode ridge

    # Sin bottleneck (ablation)
    uv run python train.py --subject sub-01 --mode no_bottleneck

    # Sin HRF delay (ablation)
    uv run python train.py --subject sub-01 --mode no_hrf

    # Todos los sujetos, modo multi-sujeto
    uv run python train.py --all_subjects --mode full

DEPENDENCIAS
------------
• torch — backend de entrenamiento
• sklearn — RidgeCV para baseline lineal
• numpy — operaciones numéricas

AUTOR
-----
Proyecto GemmaBe — Brain Encoding con Gemma 4 E2B-it.
================================================================================
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import ModelConfig
from src.tail_model import TailModel
from src.dataset import PreExtractedDataset, MultiSubjectDataset, WindowedDataset, collate_multi_subject


# =============================================================================
# Métricas
# =============================================================================

def pearson_per_vertex(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calcula la correlación de Pearson por parcela/vértice.

    Esta es la métrica estándar en brain encoding (TriBE v2, Algonauts, etc.).
    Es invariante a escala y offset, lo cual es crucial porque la señal BOLD
    no tiene unidades absolutas.

    Args:
        predicted: (T, N) — predicciones BOLD a lo largo del tiempo.
        target:    (T, N) — BOLD real.

    Returns:
        correlations: (N,) — r de Pearson por parcela. Valores en [-1, 1].
    """
    p = predicted - predicted.mean(dim=0, keepdim=True)
    t = target - target.mean(dim=0, keepdim=True)
    numerator = (p * t).sum(dim=0)
    denominator = p.norm(dim=0) * t.norm(dim=0) + 1e-8
    return numerator / denominator


# =============================================================================
# Temporal Transformer Training Loop
# =============================================================================

def train_temporal_model(
    model: nn.Module,
    dataset: WindowedDataset,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 3e-5,
    val_split: float = 0.2,
) -> dict:
    """
    Loop de entrenamiento para el modo temporal (Transformer).

    A diferencia de train_pytorch_model, este loop:
    1. Recibe ventanas (B, W, 1536) en lugar de TRs individuales (B, 1536).
    2. Calcula MSE loss sobre TODOS los timesteps de la ventana.
    3. Para validación, colapsa todas las ventanas para calcular Pearson global.

    Args:
        model:       TailModel con temporal=True.
        dataset:     WindowedDataset con ventanas de TRs.
        device:      torch.device para entrenamiento.
        epochs:      Número de épocas.
        batch_size:  Número de ventanas por batch (ojo: cada ventana es W TRs).
        lr:          Learning rate (más bajo que pointwise por la mayor capacidad).
        val_split:   Fracción de datos para validación.

    Returns:
        dict con history, pearson_map, y best_val_loss.
    """
    total = len(dataset)
    val_size = int(total * val_split)
    train_size = total - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Batch size más pequeño: cada ventana tiene W timesteps
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    effective_samples = train_size * dataset.window_size
    print(f"  Train: {train_size} ventanas ({effective_samples:,} TRs efectivos)")
    print(f"  Val:   {val_size} ventanas")

    model = model.to(device).float()
    if hasattr(model, 'print_parameter_summary'):
        model.print_parameter_summary()

    # LR más bajo y weight decay más fuerte para regularizar el Transformer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine annealing: baja el LR gradualmente durante el entrenamiento
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_pearson": []}

    start_time = time.time()

    for epoch in range(epochs):
        # TRAIN
        model.train()
        train_losses = []
        for feat_window, bold_window in train_loader:
            # feat_window: (B, W, 1536), bold_window: (B, W, 1000)
            feat_window = feat_window.to(device).float()
            bold_window = bold_window.to(device).float()

            outputs = model(feat_window)
            # Loss sobre TODOS los timesteps de la ventana
            loss = criterion(outputs["predicted_bold"].float(), bold_window.float())

            loss.backward()
            # Gradient clipping para estabilizar el Transformer
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        scheduler.step()

        # VALIDATION
        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for feat_window, bold_window in val_loader:
                feat_window = feat_window.to(device).float()
                bold_window = bold_window.to(device).float()

                outputs = model(feat_window)
                loss = criterion(outputs["predicted_bold"].float(), bold_window.float())
                val_losses.append(loss.item())

                # Colapsar ventanas: (B, W, V) → (B*W, V) para Pearson global
                B, W, V = outputs["predicted_bold"].shape
                all_preds.append(outputs["predicted_bold"].cpu().float().reshape(-1, V))
                all_targets.append(bold_window.cpu().float().reshape(-1, V))

        avg_val_loss = sum(val_losses) / len(val_losses)

        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        pearson = pearson_per_vertex(preds, targets)
        avg_pearson = pearson.mean().item()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_pearson"].append(avg_pearson)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            elapsed = time.time() - start_time
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"  Epoch {epoch+1:>3d}/{epochs} | "
                f"Train: {avg_train_loss:.6f} | "
                f"Val: {avg_val_loss:.6f} | "
                f"Pearson: {avg_pearson:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"{elapsed:.1f}s"
            )

    total_time = time.time() - start_time
    print(f"\n  Entrenamiento temporal completado en {total_time:.1f}s")
    print(f"     Mejor val loss: {best_val_loss:.6f} (epoch {best_epoch + 1})")
    print(f"     Pearson final: {history['val_pearson'][-1]:.4f}")
    print(f"     Parcelas con r > 0.15: {(pearson > 0.15).sum().item()} / {len(pearson)}")

    return {
        "model": model,
        "history": history,
        "pearson_map": pearson,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }


# =============================================================================
# Modelos auxiliares para ablation studies
# =============================================================================

class DirectLinearModel(nn.Module):
    """
    Modelo de ablation: proyección lineal directa sin bottleneck.

    En vez de Bottleneck(1536→512) + SubjectBlock(512→1000), esta clase
    implementa un único Linear(1536→num_vertices). Esto permite medir
    cuánto aporta la compresión intermedia de 512 dimensiones.
    """
    def __init__(self, input_size: int, num_vertices: int):
        super().__init__()
        self.linear = nn.Linear(input_size, num_vertices)

    def forward(self, x):
        return {"predicted_bold": self.linear(x), "compressed": x}


# =============================================================================
# Ridge Regression Baseline (sklearn)
# =============================================================================

def train_ridge(
    features_dir: Path,
    subject_id: str,
    hrf_delay: float,
    fmri_tr: float,
    val_split: float = 0.2,
) -> dict:
    """
    Entrena un Ridge Regression como baseline lineal.

    Args:
        features_dir: Directorio con real_stimulus_features.pt y fmri/.
        subject_id:   ID del sujeto.
        hrf_delay:    Retraso hemodinámico en segundos.
        fmri_tr:      TR del fMRI en segundos.
        val_split:    Fracción para validación.

    Returns:
        dict con métricas y paths a archivos guardados.
    """
    from sklearn.linear_model import RidgeCV

    print(f"\n{'='*60}")
    print(f"📐 Ridge Regression — {subject_id}")
    print(f"{'='*60}")

    dataset = PreExtractedDataset(
        features_path=str(features_dir / "real_stimulus_features.pt"),
        bold_path=str(features_dir / "fmri" / f"{subject_id}.pt"),
        hrf_delay=hrf_delay,
        fmri_tr=fmri_tr,
        normalize_bold=True,
    )

    print(f"  Muestras alineadas: {len(dataset)}")
    print(f"  Features: {dataset.features.shape}")
    print(f"  BOLD:     {dataset.bold.shape}")

    # Train/val split
    total = len(dataset)
    val_size = int(total * val_split)
    train_size = total - val_size
    train_idx, val_idx = torch.utils.data.random_split(
        range(total), [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    X_train = dataset.features[train_idx.indices].numpy()
    y_train = dataset.bold[train_idx.indices].numpy()
    X_val = dataset.features[val_idx.indices].numpy()
    y_val = dataset.bold[val_idx.indices].numpy()

    # Entrenar Ridge con validación cruzada de alpha
    print("  Entrenando RidgeCV (alphas: 0.1, 1, 10, 100, 1000, 10000)...")
    model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0], cv=3)
    model.fit(X_train, y_train)
    print(f"  Mejor alpha: {model.alpha_}")

    # Predicción en val
    y_pred = torch.tensor(model.predict(X_val), dtype=torch.float32)
    y_true = torch.tensor(y_val, dtype=torch.float32)

    pearson = pearson_per_vertex(y_pred, y_true)
    avg_pearson = pearson.mean().item()

    print(f"  ✅ Pearson promedio: {avg_pearson:.4f}")
    print(f"     (r > 0.15 en {(pearson > 0.15).sum().item()} / {len(pearson)} parcelas)")

    return {
        "pearson_map": pearson,
        "avg_pearson": avg_pearson,
        "best_alpha": model.alpha_,
    }


# =============================================================================
# PyTorch Training Loop (full, no_bottleneck, no_hrf)
# =============================================================================

def train_pytorch_model(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    val_split: float = 0.2,
) -> dict:
    """
    Loop de entrenamiento genérico para cualquier modelo PyTorch.

    Args:
        model:       nn.Module a entrenar (TailModel o DirectLinearModel).
        dataset:     PyTorch Dataset con features y BOLD alineados.
        device:      torch.device para entrenamiento.
        epochs:      Número de épocas.
        batch_size:  Tamaño de batch.
        lr:          Learning rate.
        val_split:   Fracción de datos para validación.

    Returns:
        dict con history, pearson_map, y best_val_loss.
    """
    total = len(dataset)
    val_size = int(total * val_split)
    train_size = total - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"  Train: {train_size} muestras | Val: {val_size} muestras")

    model = model.to(device).float()
    if hasattr(model, 'print_parameter_summary'):
        model.print_parameter_summary()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_pearson": []}

    start_time = time.time()

    for epoch in range(epochs):
        # TRAIN
        model.train()
        train_losses = []
        for features, bold in train_loader:
            features = features.to(device).float()
            bold = bold.to(device).float()

            outputs = model(features)
            loss = criterion(outputs["predicted_bold"].float(), bold.float())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)

        # VALIDATION
        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for features, bold in val_loader:
                features = features.to(device).float()
                bold = bold.to(device).float()

                outputs = model(features)
                loss = criterion(outputs["predicted_bold"].float(), bold.float())
                val_losses.append(loss.item())

                all_preds.append(outputs["predicted_bold"].cpu().float())
                all_targets.append(bold.cpu().float())

        avg_val_loss = sum(val_losses) / len(val_losses)

        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        pearson = pearson_per_vertex(preds, targets)
        avg_pearson = pearson.mean().item()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_pearson"].append(avg_pearson)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            elapsed = time.time() - start_time
            print(
                f"  Epoch {epoch+1:>3d}/{epochs} | "
                f"Train: {avg_train_loss:.6f} | "
                f"Val: {avg_val_loss:.6f} | "
                f"Pearson: {avg_pearson:.4f} | "
                f"⏱️ {elapsed:.1f}s"
            )

    total_time = time.time() - start_time
    print(f"\n  ✅ Entrenamiento completado en {total_time:.1f}s")
    print(f"     Mejor val loss: {best_val_loss:.6f} (epoch {best_epoch + 1})")
    print(f"     Pearson final: {history['val_pearson'][-1]:.4f}")
    print(f"     Parcelas con r > 0.15: {(pearson > 0.15).sum().item()} / {len(pearson)}")

    return {
        "model": model,
        "history": history,
        "pearson_map": pearson,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }


# =============================================================================
# Entrenamiento single-subject con dispatch por modo
# =============================================================================

def train_single_subject(
    features_dir: str,
    fmri_dir: str,
    subject_id: str,
    config: ModelConfig,
    mode: str = "full",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    val_split: float = 0.2,
    fmri_tr: float = 1.49,
) -> dict:
    """
    Entrena un modelo para un único sujeto, despachando según el modo.

    Args:
        features_dir: Directorio con real_stimulus_features.pt.
        fmri_dir:     Directorio con archivos sub-XX.pt de BOLD.
        subject_id:   ID del sujeto.
        config:       ModelConfig.
        mode:         "full", "ridge", "no_bottleneck", o "no_hrf".
        epochs:       Número de épocas (no aplica a ridge).
        batch_size:   Tamaño de batch.
        lr:           Learning rate.
        val_split:    Fracción de validación.
        fmri_tr:      TR del fMRI en segundos.

    Returns:
        dict con resultados y pearson_map.
    """
    features_dir = Path(features_dir)
    fmri_dir = Path(fmri_dir)
    device = config.device

    print(f"\n{'='*60}")
    print(f"🧠 Entrenando {subject_id} | modo: {mode}")
    print(f"{'='*60}")

    # Calcular HRF delay según modo
    hrf_delay = 0.0 if mode == "no_hrf" else config.hrf_delay_seconds

    # --- Cargar dataset ---
    bold_path = fmri_dir / f"{subject_id}.pt"
    if not bold_path.exists():
        raise FileNotFoundError(f"No se encontró BOLD para {subject_id}: {bold_path}")

    dataset = PreExtractedDataset(
        features_path=str(features_dir / "real_stimulus_features.pt"),
        bold_path=str(bold_path),
        hrf_delay=hrf_delay,
        fmri_tr=fmri_tr,
        normalize_bold=True,
    )

    delay_in_trs = int(hrf_delay / fmri_tr)
    print(f"\n🔍 Alineación:")
    print(f"  • HRF delay: {hrf_delay}s ({delay_in_trs} TRs)")
    print(f"  • Muestras alineadas: {len(dataset)}")
    print(f"  • Features: {dataset.features.shape}")
    print(f"  • BOLD:     {dataset.bold.shape}")

    # Auto-detectar num_vertices
    detected_voxels = dataset.bold.shape[1]
    if config.num_vertices == 0 or config.num_vertices != detected_voxels:
        print(f"  🔍 num_vertices auto-detectado: {detected_voxels:,}")
        config.num_vertices = detected_voxels

    # --- Dispatch por modo ---
    if mode == "ridge":
        results = train_ridge(
            features_dir=features_dir,
            subject_id=subject_id,
            hrf_delay=hrf_delay,
            fmri_tr=fmri_tr,
            val_split=val_split,
        )
        model_for_save = None  # RidgeCV no se guarda como .pt fácilmente

    elif mode == "no_bottleneck":
        model = DirectLinearModel(input_size=config.gemma_hidden_size, num_vertices=config.num_vertices)
        results = train_pytorch_model(
            model=model, dataset=dataset, device=device,
            epochs=epochs, batch_size=batch_size, lr=lr, val_split=val_split,
        )
        model_for_save = model

    elif mode == "temporal":
        # --- Modo Temporal: Transformer entre Bottleneck y SubjectBlock ---
        windowed_dataset = WindowedDataset(
            features_path=str(features_dir / "real_stimulus_features.pt"),
            bold_path=str(bold_path),
            window_size=config.window_size_trs,
            stride=1,  # máximo solapamiento = más datos de entrenamiento
            hrf_delay=hrf_delay,
            fmri_tr=fmri_tr,
            normalize_bold=True,
        )
        model = TailModel(config=config, temporal=True)
        # Batch size más pequeño y LR más bajo para el Transformer
        temporal_batch = max(4, batch_size // 8)
        temporal_lr = lr * 0.3  # ~3e-5 si lr=1e-4
        results = train_temporal_model(
            model=model, dataset=windowed_dataset, device=device,
            epochs=epochs, batch_size=temporal_batch, lr=temporal_lr,
            val_split=val_split,
        )
        model_for_save = results["model"]

    else:  # "full" o "no_hrf"
        model = TailModel(config=config)
        results = train_pytorch_model(
            model=model, dataset=dataset, device=device,
            epochs=epochs, batch_size=batch_size, lr=lr, val_split=val_split,
        )
        model_for_save = results.get("model", model)

    # --- Guardar resultados ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results") / f"{mode}_{subject_id}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Guardar Pearson map
    torch.save(results["pearson_map"], out_dir / "pearson_map.pt")

    # Guardar history (si existe)
    if "history" in results:
        with open(out_dir / "history.json", 'w') as f:
            json.dump(results["history"], f, indent=2)

    # Guardar modelo (si es PyTorch)
    if model_for_save is not None:
        torch.save(model_for_save.state_dict(), out_dir / "model.pt")

    # Guardar config
    config_data = {
        "mode": mode,
        "subject_id": subject_id,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "hrf_delay": hrf_delay,
        "fmri_tr": fmri_tr,
        "num_vertices": config.num_vertices,
    }
    if mode == "temporal":
        config_data.update({
            "window_size_trs": config.window_size_trs,
            "transformer_layers": config.transformer_layers,
            "transformer_heads": config.transformer_heads,
            "transformer_dropout": config.transformer_dropout,
        })
    with open(out_dir / "config.json", 'w') as f:
        json.dump(config_data, f, indent=2)

    print(f"\n  💾 Resultados guardados en: {out_dir}")
    return results


# =============================================================================
# Multi-subject training
# =============================================================================

def train_multi_subject(
    features_dir: str,
    fmri_dir: str,
    subject_ids: list[str],
    config: ModelConfig,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
    fmri_tr: float = 1.49,
) -> dict:
    """
    Entrena el TailModel con múltiples sujetos simultáneamente.

    El Bottleneck es compartido entre sujetos, pero cada uno tiene su
    SubjectBlock propio. Esto fuerza al modelo a aprender una representación
    común que funcione para todos los cerebros.
    """
    features_dir = Path(features_dir)
    fmri_dir = Path(fmri_dir)
    device = config.device

    print(f"\n{'='*60}")
    print(f"👥 Multi-sujeto: {', '.join(subject_ids)}")
    print(f"{'='*60}")

    dataset = MultiSubjectDataset(
        features_path=str(features_dir / "real_stimulus_features.pt"),
        bold_dir=str(fmri_dir),
        subject_ids=subject_ids,
        hrf_delay=config.hrf_delay_seconds,
        normalize_bold=True,
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_multi_subject,
    )

    model = TailModel(config=config, subject_ids=subject_ids)
    model = model.to(device).float()
    model.print_parameter_summary()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for batch in dataloader:
            total_loss = 0
            for subject_id, features, bold in batch:
                features = features.to(device).float()
                bold = bold.to(device).float()
                outputs = model(features, subject_id=subject_id)
                loss = criterion(outputs["predicted_bold"].float(), bold.float())
                total_loss = total_loss + loss

            avg_loss = total_loss / len(batch)
            avg_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_losses.append(avg_loss.item())

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1:>3d}/{epochs} | Loss: {avg_epoch_loss:.6f} | ⏱️ {elapsed:.1f}s")

    total_time = time.time() - start_time
    print(f"\n  ✅ Multi-sujeto completado en {total_time:.1f}s")
    return {"final_loss": avg_epoch_loss}


# =============================================================================
# Entry point — CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento offline de GemmaBE con múltiples modos"
    )
    parser.add_argument("--features_dir", type=str, default="./data/features",
                        help="Directorio con real_stimulus_features.pt")
    parser.add_argument("--fmri_dir", type=str, default=None,
                        help="Directorio con archivos sub-XX.pt de BOLD. "
                             "Si no se especifica, usa features_dir/fmri/")
    parser.add_argument("--subject", nargs="+", default=None,
                        help="IDs de sujeto. Ej: --subject sub-01 sub-02")
    parser.add_argument("--all_subjects", action="store_true",
                        help="Usar todos los sujetos encontrados en fmri_dir/")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "temporal", "ridge", "no_bottleneck", "no_hrf"],
                        help="Modo de entrenamiento (default: full). 'temporal' usa Transformer.")
    parser.add_argument("--window_size", type=int, default=None,
                        help="Tamaño de ventana en TRs para modo temporal (default: 67 ≈ 100s)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fmri_tr", type=float, default=1.49,
                        help="TR del fMRI en segundos (1.49 para Algonauts 2025)")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")

    args = parser.parse_args()

    config = ModelConfig()
    if args.window_size is not None:
        config.window_size_trs = args.window_size
    if args.device != "auto":
        config.override_device(args.device)

    features_dir = Path(args.features_dir)
    fmri_dir = Path(args.fmri_dir) if args.fmri_dir else features_dir / "fmri"

    # Determinar sujetos
    if args.all_subjects:
        subject_ids = sorted([f.stem for f in fmri_dir.glob("sub-*.pt")])
        print(f"  Encontrados {len(subject_ids)} sujetos: {', '.join(subject_ids)}")
    elif args.subject:
        subject_ids = args.subject
    else:
        print("Error: Especifica --subject o --all_subjects")
        return

    # Entrenar
    if len(subject_ids) == 1:
        train_single_subject(
            features_dir=str(features_dir),
            fmri_dir=str(fmri_dir),
            subject_id=subject_ids[0],
            config=config,
            mode=args.mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            fmri_tr=args.fmri_tr,
        )
    else:
        if args.mode != "full":
            print("⚠️  Multi-sujeto solo soporta modo 'full'. Usando full.")
        train_multi_subject(
            features_dir=str(features_dir),
            fmri_dir=str(fmri_dir),
            subject_ids=subject_ids,
            config=config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            fmri_tr=args.fmri_tr,
        )


if __name__ == "__main__":
    main()
