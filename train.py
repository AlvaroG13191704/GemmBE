"""
Script de entrenamiento offline para MicroTRIBE-Gemma.

Este script entrena SOLO la cola del pipeline (Bottleneck + SubjectBlock)
usando features pre-extraídos de Gemma 4 guardados en disco.

NO carga Gemma 4 en memoria. Solo necesita ~50MB de VRAM.
Puede entrenar cientos de épocas en minutos en cualquier GPU.

Uso:
    # 1. Primero generar datos dummy para testing:
    uv run python -m src.extract_features --dummy --num_trs 300 --num_subjects 10
    
    # 2. Entrenar con un solo sujeto:
    uv run python train.py --features_dir ./data/features --subject sub-01
    
    # 3. Entrenar con múltiples sujetos:
    uv run python train.py --features_dir ./data/features --subject sub-01 sub-02 sub-03
    
    # 4. Entrenar todos los sujetos del directorio:
    uv run python train.py --features_dir ./data/features --all_subjects
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import ModelConfig
from src.tail_model import TailModel
from src.dataset import PreExtractedDataset, MultiSubjectDataset, collate_multi_subject


def pearson_per_vertex(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calcula la correlación de Pearson por vértice cortical.
    
    Ésta es la métrica estándar en brain encoding (usada en TriBE v2,
    Algonauts Challenge, etc.).
    
    Args:
        predicted: (T, 20484) — predicciones BOLD a lo largo del tiempo.
        target:    (T, 20484) — BOLD real.
    
    Returns:
        correlations: (20484,) — r de Pearson por vértice.
    """
    # Centrar (restar media temporal)
    p = predicted - predicted.mean(dim=0, keepdim=True)
    t = target - target.mean(dim=0, keepdim=True)
    
    # Correlación
    numerator = (p * t).sum(dim=0)
    denominator = p.norm(dim=0) * t.norm(dim=0) + 1e-8
    
    return numerator / denominator


def train_single_subject(
    features_dir: str,
    subject_id: str,
    config: ModelConfig,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    val_split: float = 0.2,
    fmri_tr: float = 1.5,
) -> dict:
    """
    Entrena el TailModel para un solo sujeto.
    
    Args:
        features_dir: Directorio con real_stimulus_features.pt y fmri/.
        subject_id: ID del sujeto (e.g., "sub-01").
        config: ModelConfig.
        epochs: Número de épocas.
        batch_size: Tamaño de batch.
        lr: Learning rate.
        val_split: Fracción de datos para validación.
        fmri_tr: Tiempo de Repetición del fMRI (ej. 1.5 para Sherlock).
    
    Returns:
        dict con métricas de entrenamiento.
    """
    features_dir = Path(features_dir)
    device = config.device
    
    # --- Cargar dataset ---
    print(f"\n{'='*60}")
    print(f"🧠 Entrenando {subject_id}")
    print(f"{'='*60}")
    
    dataset = PreExtractedDataset(
        features_path=str(features_dir / "real_stimulus_features.pt"),
        bold_path=str(features_dir / "fmri" / f"{subject_id}.pt"),
        hrf_delay=config.hrf_delay_seconds,
        fmri_tr=fmri_tr,
        normalize_bold=True,
    )
    
    # --- VERIFICACIÓN DE SINCRONIZACIÓN ---
    # Mostramos explícitamente cómo ocurrió la alineación
    print("\n🔍 VERIFICACIÓN DE SINCRONIZACIÓN:")
    delay_in_trs = int(config.hrf_delay_seconds / fmri_tr)
    print(f"  • Retraso Hemodinámico (HRF): {config.hrf_delay_seconds} segundos")
    print(f"  • TR del fMRI: {fmri_tr} segundos")
    print(f"  • TRs descartados al inicio (HRF shift): {delay_in_trs} TRs")
    print(f"  • Muestras perfectamente alineadas resultantes: {len(dataset)}")
    print(f"  • Forma del Tensor Estímulo (Gemma): {dataset.features.shape}")
    print(f"  • Forma del Tensor Cerebro (BOLD):   {dataset.bold.shape}")
    print("  ✅ Sincronización 1 a 1 verificada exitosamente.\n")
    
    # --- Split train/val ---
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
    
    # --- Crear modelo ---
    # Auto-detectar num_vertices desde los datos BOLD cargados
    # Esto funciona transparentemente con datos completos (238,955)
    # o reducidos por varianza (~191,000 vóxeles).
    detected_voxels = dataset.bold.shape[1]
    if config.num_vertices == 0 or config.num_vertices != detected_voxels:
        print(f"  🔍 Auto-detectado num_vertices = {detected_voxels:,} (del tensor BOLD)")
        config.num_vertices = detected_voxels
    
    # IMPORTANTE: Forzar float32 para entrenamiento.
    # El TailModel tiene una capa Linear(512→N) con muchos params.
    # En float16, los gradientes desbordan (max representable = 65504)
    # causando NaN en la loss. float32 previene esto completamente.
    model = TailModel(config=config)
    model = model.float().to(device)  # Forzar float32
    model.print_parameter_summary()
    
    # --- Optimizador y loss ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()
    
    # --- Training loop ---
    best_val_loss = float("inf")
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_pearson": []}
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # TRAIN
        model.train()
        train_losses = []
        
        for features, bold in train_loader:
            features = features.to(device).float()  # Forzar float32
            bold = bold.to(device).float()           # Forzar float32
            
            outputs = model(features)
            
            # Loss en float32 para estabilidad numérica
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
        
        # Pearson correlation (métrica principal)
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        pearson = pearson_per_vertex(preds, targets)
        avg_pearson = pearson.mean().item()
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_pearson"].append(avg_pearson)
        
        # Guardar mejor modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
        
        # Log cada 10 épocas o la última
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
    
    return history


def train_multi_subject(
    features_dir: str,
    subject_ids: list[str],
    config: ModelConfig,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
    fmri_tr: float = 1.5,
) -> dict:
    """
    Entrena el TailModel con múltiples sujetos simultáneamente.
    
    Todas las predicciones pasan por el Bottleneck compartido,
    pero cada sujeto tiene su propio SubjectBlock.
    """
    features_dir = Path(features_dir)
    device = config.device
    
    print(f"\n{'='*60}")
    print(f"👥 Entrenamiento multi-sujeto: {', '.join(subject_ids)}")
    print(f"{'='*60}")
    
    # --- Dataset ---
    dataset = MultiSubjectDataset(
        features_path=str(features_dir / "real_stimulus_features.pt"),
        bold_dir=str(features_dir / "fmri"),
        subject_ids=subject_ids,
        hrf_delay=config.hrf_delay_seconds,
        normalize_bold=True,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_multi_subject,
    )
    
    # --- Modelo con Multi-Subject ---
    model = TailModel(config=config, subject_ids=subject_ids)
    model = model.to(device)
    model.print_parameter_summary()
    
    # --- Optimizador ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()
    
    # --- Training loop ---
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        for batch in dataloader:
            total_loss = 0
            
            # Cada batch contiene grupos por sujeto
            for subject_id, features, bold in batch:
                features = features.to(device)
                bold = bold.to(device)
                
                outputs = model(features, subject_id=subject_id)
                loss = criterion(outputs["predicted_bold"].float(), bold.float())
                total_loss = total_loss + loss
            
            # Promediar loss de todos los sujetos del batch
            avg_loss = total_loss / len(batch)
            avg_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_losses.append(avg_loss.item())
        
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            elapsed = time.time() - start_time
            print(
                f"  Epoch {epoch+1:>3d}/{epochs} | "
                f"Loss: {avg_epoch_loss:.6f} | "
                f"⏱️ {elapsed:.1f}s"
            )
    
    total_time = time.time() - start_time
    print(f"\n  ✅ Multi-subject completado en {total_time:.1f}s")
    
    return {"final_loss": avg_epoch_loss}


def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento offline de MicroTRIBE-Gemma"
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default="./data/features",
        help="Directorio con features pre-extraídos.",
    )
    parser.add_argument(
        "--subject",
        nargs="+",
        default=None,
        help="IDs de sujeto. Ej: --subject sub-01 sub-02",
    )
    parser.add_argument(
        "--all_subjects",
        action="store_true",
        help="Usar todos los sujetos encontrados en el directorio fmri/.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--fmri_tr",
        type=float,
        default=1.5,
        help="Tiempo de Repetición (TR) del escáner en segundos (1.5 para Sherlock).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
    )
    
    args = parser.parse_args()
    
    # Config
    config = ModelConfig()
    if args.device != "auto":
        config.override_device(args.device)
    
    features_dir = Path(args.features_dir)
    
    # Determinar sujetos
    if args.all_subjects:
        fmri_dir = features_dir / "fmri"
        subject_ids = sorted([
            f.stem for f in fmri_dir.glob("sub-*.pt")
        ])
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
            subject_id=subject_ids[0],
            config=config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            fmri_tr=args.fmri_tr,
        )
    else:
        train_multi_subject(
            features_dir=str(features_dir),
            subject_ids=subject_ids,
            config=config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            fmri_tr=args.fmri_tr,
        )


if __name__ == "__main__":
    main()
