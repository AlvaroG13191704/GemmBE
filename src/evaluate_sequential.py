"""
evaluate_sequential.py — Evaluación secuencial en los últimos 20k TRs.

Carga un modelo entrenado y evalúa en los últimos ~20k TRs del dataset,
representando episodios finales de Friends que el modelo nunca vio
en secuencia durante entrenamiento.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# pyrefly: ignore [missing-import]
from src.models.temporal_full_model import TemporalFullModel
# pyrefly: ignore [missing-import]
from src.utils.temporal_alignment import HRFAligner


class _SlidingWindowDataset(Dataset):
    """Dataset de ventanas deslizantes (on-the-fly)."""

    def __init__(self, features, bold, window_size=67, stride=5):
        self.features = features
        self.bold = bold
        self.window_size = window_size
        self.stride = stride
        self.num_windows = max(0, (features.shape[0] - window_size) // stride + 1)

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        return self.features[start:end], self.bold[start:end]


def compute_pearson(pred, target):
    """Pearson correlation per parcel."""
    p = pred - pred.mean(dim=0, keepdim=True)
    t = target - target.mean(dim=0, keepdim=True)
    num = (p * t).sum(dim=0)
    den = p.norm(dim=0) * t.norm(dim=0) + 1e-8
    return num / den


def evaluate_sequential(
    checkpoint_path: str,
    features_path: str,
    bold_path: str,
    num_test_trs: int = 20000,
    window_size: int = 67,
    stride: int = 5,
    batch_size: int = 16,
    hrf_delay: float = 5.0,
    fmri_tr: float = 1.49,
    device: str = "mps",
):
    """Evalúa secuencialmente en los últimos N TRs."""

    print("=" * 60)
    print("Evaluación Secuencial")
    print("=" * 60)

    # ─── 1. Cargar modelo ──────────────────────────────────────────────────
    print(f"\nCargando modelo: {checkpoint_path}")
    # Manual load to avoid hyperparameter conflicts
    import lightning as L
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    
    # Cast all tensors in state_dict to float32 (checkpoint was saved in bfloat16)
    state_dict = {k: v.float() if v.dtype in (torch.bfloat16, torch.float16) else v for k, v in state_dict.items()}
    
    # Create model with default params
    model = TemporalFullModel(
        stimulus_type="multimodal",
        subject_id="sub-01",
        num_vertices=1000,
        lr=1e-4,
        weight_decay=1e-5,
        max_epochs=100,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"  Modelo cargado: {model.model_name}")
    print(f"  Device: {device}")

    # ─── 2. Cargar datos ───────────────────────────────────────────────────
    print(f"\nCargando datos:")
    print(f"  Features: {features_path}")
    print(f"  BOLD:     {bold_path}")

    features = torch.load(features_path, weights_only=True)
    bold = torch.load(bold_path, weights_only=True)
    
    # Cast features to float32 (saved in bfloat16 from extraction)
    features = features.to(torch.float32)
    
    print(f"  Features shape: {features.shape}, dtype: {features.dtype}")
    print(f"  BOLD shape:     {bold.shape}, dtype: {bold.dtype}")

    # ─── 3. Alineación HRF ─────────────────────────────────────────────────
    aligner = HRFAligner(hrf_delay_seconds=hrf_delay, fmri_tr_seconds=fmri_tr)
    features, bold = aligner.align_stimulus_to_fmri(features, bold)
    print(f"  After HRF align: features={features.shape}, bold={bold.shape}")

    # ─── 4. Normalización z-score (igual que entrenamiento) ────────────────
    mean = bold.mean(dim=0, keepdim=True)
    std = bold.std(dim=0, keepdim=True).clamp(min=1e-8)
    bold = (bold - mean) / std
    print(f"  BOLD normalized (z-score per parcel)")

    # ─── 5. Tomar últimos N TRs como test secuencial ────────────────────────
    total_trs = features.shape[0]
    start_idx = max(0, total_trs - num_test_trs)

    test_features = features[start_idx:]
    test_bold = bold[start_idx:]

    actual_test_trs = test_features.shape[0]
    print(f"\n  Total TRs: {total_trs}")
    print(f"  Test TRs:  {actual_test_trs} (from {start_idx} to end)")
    print(f"  Test %:    {actual_test_trs/total_trs*100:.1f}%")

    # ─── 6. Crear ventanas deslizantes ─────────────────────────────────────
    dataset = _SlidingWindowDataset(test_features, test_bold, window_size, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"  Windows created: {len(dataset)} (window={window_size}, stride={stride})")

    # ─── 7. Inferencia ─────────────────────────────────────────────────────
    all_pred = []
    all_bold = []

    print(f"\nRunning inference...")
    with torch.no_grad():
        for batch_idx, (feat_batch, bold_batch) in enumerate(dataloader):
            feat_batch = feat_batch.to(device)
            pred_batch = model(feat_batch)
            all_pred.append(pred_batch.cpu())
            all_bold.append(bold_batch.cpu())

            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}/{len(dataloader)}")

    # ─── 8. Concatenar y aplanar ─────────────────────────────────────────
    all_pred = torch.cat(all_pred)  # (N, W, 1000)
    all_bold = torch.cat(all_bold)  # (N, W, 1000)

    # Aplanar: (N*W, 1000)
    pred_flat = all_pred.reshape(-1, all_pred.shape[-1])
    bold_flat = all_bold.reshape(-1, all_bold.shape[-1])

    print(f"\n  Predictions shape: {pred_flat.shape}")
    print(f"  BOLD shape:        {bold_flat.shape}")

    # ─── 9. Calcular métricas ──────────────────────────────────────────────
    mse = nn.functional.mse_loss(pred_flat, bold_flat).item()
    pearson = compute_pearson(pred_flat, bold_flat)
    pearson_mean = pearson.mean().item()
    pearson_std = pearson.std().item()
    pearson_median = pearson.median().item()
    n_significant = (pearson > 0.15).sum().item()

    print(f"\n{'=' * 60}")
    print("RESULTADOS SECUENCIALES")
    print(f"{'=' * 60}")
    print(f"  MSE:              {mse:.6f}")
    print(f"  Pearson mean:     {pearson_mean:.4f}")
    print(f"  Pearson std:      {pearson_std:.4f}")
    print(f"  Pearson median:   {pearson_median:.4f}")
    print(f"  Pearson > 0.15:   {n_significant}/{len(pearson)} ({n_significant/len(pearson)*100:.1f}%)")
    print(f"  Pearson min:      {pearson.min():.4f}")
    print(f"  Pearson max:      {pearson.max():.4f}")
    print(f"{'=' * 60}")

    return {
        "mse": mse,
        "pearson_mean": pearson_mean,
        "pearson_std": pearson_std,
        "pearson_median": pearson_median,
        "pearson>0.15": n_significant,
        "test_trs": actual_test_trs,
        "windows": len(dataset),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluación secuencial")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path al .ckpt")
    parser.add_argument("--features", type=str, default="data/features/real_stimulus_features.pt")
    parser.add_argument("--bold", type=str, default="data/subjects_fmri_filtered/sub-01.pt")
    parser.add_argument("--test_trs", type=int, default=20000, help="Número de TRs de test")
    parser.add_argument("--window_size", type=int, default=67)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    results = evaluate_sequential(
        checkpoint_path=args.checkpoint,
        features_path=args.features,
        bold_path=args.bold,
        num_test_trs=args.test_trs,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        device=args.device,
    )

    print("\nEvaluación completada.")


if __name__ == "__main__":
    main()
