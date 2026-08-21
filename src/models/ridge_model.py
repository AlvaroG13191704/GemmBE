"""
Ridge Regression Baseline para Brain Encoding.

Implementación de Ridge regression con StandardScaler como baseline
determinístico. Usa RidgeCV para selección automática de alpha.

Comparación con modelos neuronales:
    - Ridge: modelo lineal, no captura interacciones no lineales
    - TemporalFullModel: Transformer, captura dependencias temporales
    - WithoutTemporalFullModel: MLP pointwise, captura no linealidades

Si Ridge performa similar a modelos complejos, significa que los features
de Gemma 4 ya contienen la información relevante en forma lineal.
"""

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


def train_ridge_baseline(
    split_dir: str = "data/train_test_split",
    subject_id: str = "sub-01",
    alphas: list = None,
    save_dir: str = "results/ridge_baseline",
) -> dict:
    """
    Entrena Ridge regression como baseline y evalúa en Season 6 hold-out.

    Args:
        split_dir: Directorio con features_train.pt, features_test.pt, etc.
        subject_id: Sujeto a evaluar (sub-01 o sub-02).
        alphas: Lista de alphas para RidgeCV. Default: logspace(-1, 4, 20).
        save_dir: Directorio para guardar resultados.

    Returns:
        dict con métricas de test.
    """
    if alphas is None:
        alphas = np.logspace(-1, 4, 20).tolist()

    split_path = Path(split_dir)
    output_dir = Path(save_dir) / f"ridge_{subject_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Ridge Baseline — {subject_id}")
    print(f"{'='*60}")

    # ─── Cargar datos ──────────────────────────────────────────────────────────
    features_train = torch.load(
        split_path / "features_train.pt", weights_only=True
    ).float().numpy()
    features_test = torch.load(
        split_path / "features_test.pt", weights_only=True
    ).float().numpy()
    bold_train = torch.load(
        split_path / f"bold_train_{subject_id}.pt", weights_only=True
    ).float().numpy()
    bold_test = torch.load(
        split_path / f"bold_test_{subject_id}.pt", weights_only=True
    ).float().numpy()

    print(f"Features train: {features_train.shape}")
    print(f"Features test:  {features_test.shape}")
    print(f"BOLD train:     {bold_train.shape}")
    print(f"BOLD test:      {bold_test.shape}")

    # Alinear si hay diferencia de TRs (por truncamiento de fMRI)
    min_train = min(features_train.shape[0], bold_train.shape[0])
    features_train = features_train[:min_train]
    bold_train = bold_train[:min_train]

    min_test = min(features_test.shape[0], bold_test.shape[0])
    features_test = features_test[:min_test]
    bold_test = bold_test[:min_test]

    print(f"\nDespués de alineamiento:")
    print(f"  Train: {features_train.shape[0]} TRs")
    print(f"  Test:  {features_test.shape[0]} TRs")

    # ─── Escalar features ────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(features_train)
    X_test = scaler.transform(features_test)

    print(f"\nFeatures escalados (StandardScaler)")

    # ─── Entrenar RidgeCV ──────────────────────────────────────────────────────
    print(f"\nEntrenando RidgeCV con alphas: {alphas[0]:.1f} ... {alphas[-1]:.1f}")
    ridge = RidgeCV(alphas=alphas)
    ridge.fit(X_train, bold_train)

    print(f"  Mejor alpha: {ridge.alpha_:.4f}")
    print(f"  CV Score:    {ridge.score(X_train, bold_train):.6f}")

    # ─── Evaluar en test ───────────────────────────────────────────────────────
    pred_test = ridge.predict(X_test)

    pearsons = []
    for i in range(bold_test.shape[1]):
        r, _ = pearsonr(pred_test[:, i], bold_test[:, i])
        pearsons.append(r)
    pearsons = np.array(pearsons)

    mean_pearson = np.nanmean(pearsons)
    std_pearson = np.nanstd(pearsons)
    gt_015 = int(np.sum(pearsons > 0.15))
    gt_025 = int(np.sum(pearsons > 0.25))

    # MSE
    mse = np.mean((pred_test - bold_test) ** 2)

    print(f"\n{'='*60}")
    print(f"RESULTADOS TEST (Season 6 hold-out)")
    print(f"{'='*60}")
    print(f"  Test Pearson mean:  {mean_pearson:.4f}")
    print(f"  Test Pearson std:   {std_pearson:.4f}")
    print(f"  Test Pearson >0.15: {gt_015}/1000")
    print(f"  Test Pearson >0.25: {gt_025}/1000")
    print(f"  Test MSE:           {mse:.6f}")

    # ─── Guardar resultados ────────────────────────────────────────────────────
    results = {
        "model": "ridge_baseline",
        "subject": subject_id,
        "alpha": float(ridge.alpha_),
        "test_pearson": float(mean_pearson),
        "test_pearson_mean": float(mean_pearson),
        "test_pearson_std": float(std_pearson),
        "test_pearson>0.15": gt_015,
        "test_pearson>0.25": gt_025,
        "test_loss": float(mse),
        "train_trs": int(min_train),
        "test_trs": int(min_test),
    }

    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Guardar Pearson map
    torch.save(torch.from_numpy(pearsons), output_dir / "pearson_map_test.pt")

    # Guardar predicciones
    torch.save(torch.from_numpy(pred_test), output_dir / "predictions_test.pt")

    print(f"\n💾 Resultados guardados en: {output_dir}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ridge Regression Baseline")
    parser.add_argument("--split_dir", default="data/train_test_split")
    parser.add_argument("--subjects", nargs="+", default=["sub-01", "sub-02"])
    parser.add_argument("--save_dir", default="results/ridge_baseline")
    args = parser.parse_args()

    all_results = {}
    for subject in args.subjects:
        results = train_ridge_baseline(
            split_dir=args.split_dir,
            subject_id=subject,
            save_dir=args.save_dir,
        )
        all_results[subject] = results

    # Guardar resumen
    summary_path = Path(args.save_dir) / "ridge_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("RESUMEN RIDGE BASELINE")
    print(f"{'='*60}")
    for sub, res in all_results.items():
        print(f"  {sub}: Pearson={res['test_pearson']:.4f} (alpha={res['alpha']:.1f})")


if __name__ == "__main__":
    main()
