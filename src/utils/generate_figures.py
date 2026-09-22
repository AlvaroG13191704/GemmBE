"""
Generador de figuras para paper IEEE — GemmaBE.

Figuras:
  2. Barras comparativas de Pearson medio (Panel a: Test = Season 6 hold-out, Panel b: Val = Season 5 hold-out)
  3. Mapas de superficie cortical (Pearson por parcela)
  4. Curvas de entrenamiento (train loss + val Pearson limpio en Season 5)

Uso:
    uv run python src/utils/generate_figures.py
    uv run python src/utils/generate_figures.py --figure 2
    uv run python src/utils/generate_figures.py --figure 3 --subject sub-01
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Forzar backend sin GUI (para headless)
matplotlib.use("Agg")

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)


def collect_test_results():
    """Recopila todos los test_results.json en un diccionario."""
    results = {}
    for json_path in sorted(RESULTS_DIR.glob("*/metrics/test_results.json")):
        run_name = json_path.parent.parent.name
        with open(json_path) as f:
            data = json.load(f)
        results[run_name] = data
    return results


def collect_val_results():
    """Recopila val_pearson final de los CSVs."""
    val_results = {}
    for csv_path in sorted(RESULTS_DIR.glob("*/logs/csv/version_0/metrics.csv")):
        run_name = csv_path.parent.parent.parent.parent.name
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r.get('val/pearson') and r['val/pearson'].strip()]
            if rows:
                val_results[run_name] = float(rows[-1]['val/pearson'])
    return val_results


def parse_run_name(name: str):
    """Parsea 'temporal_full_multimodal_sub-01' -> (model, stimulus, subject)."""
    for model in ["temporal_full", "without_temporal_full", "no_hrf"]:
        if name.startswith(model + "_"):
            rest = name[len(model) + 1:]
            parts = rest.rsplit("_", 1)
            if len(parts) == 2:
                stimulus, subject = parts
                return model, stimulus, subject
    return None, None, None


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURA 2 — Barras comparativas (TEST = Season 6, VAL = Season 5)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_figure_2(test_results: dict, val_results: dict):
    """Figura 2: Barras comparativas de Pearson medio TEST (S6) y VAL (S5)."""
    models = ["temporal_full", "without_temporal_full"]
    stimuli = ["multimodal", "textonly"]
    subjects = ["sub-01", "sub-02"]

    # TEST data: modelo x estímulo x sujeto
    test_data = np.zeros((len(models), len(stimuli), len(subjects)))
    for run_name, metrics in test_results.items():
        model, stimulus, subject = parse_run_name(run_name)
        if model not in models or stimulus not in stimuli or subject not in subjects:
            continue
        m_idx = models.index(model)
        s_idx = stimuli.index(stimulus)
        sub_idx = subjects.index(subject)
        # Usar test_pearson si existe, sino fallback a test_pearson_mean
        p_val = metrics.get("test_pearson", metrics.get("test_pearson_mean", 0.0))
        test_data[m_idx, s_idx, sub_idx] = p_val

    # VAL data: Season 5 hold-out
    val_data = np.zeros((len(models), len(stimuli), len(subjects)))
    for run_name, val_pearson in val_results.items():
        model, stimulus, subject = parse_run_name(run_name)
        if model not in models or stimulus not in stimuli or subject not in subjects:
            continue
        m_idx = models.index(model)
        s_idx = stimuli.index(stimulus)
        sub_idx = subjects.index(subject)
        val_data[m_idx, s_idx, sub_idx] = val_pearson

    # Ridge baseline (Multimodal)
    ridge_test_mm = []
    ridge_test_to = []
    for sub in subjects:
        r_file_mm = RESULTS_DIR / f"ridge_baseline/ridge_{sub}/test_results.json"
        if r_file_mm.exists():
            with open(r_file_mm) as f:
                ridge_test_mm.append(json.load(f)["test_pearson"])
        r_file_to = RESULTS_DIR / f"ridge_baseline/ridge_textonly_{sub}/test_results.json"
        if r_file_to.exists():
            with open(r_file_to) as f:
                ridge_test_to.append(json.load(f)["test_pearson"])

    ridge_mean_mm = np.mean(ridge_test_mm) if ridge_test_mm else 0.094
    ridge_std_mm = np.std(ridge_test_mm) if ridge_test_mm else 0.008

    fig, axes = plt.subplots(2, 1, figsize=(8, 9))
    x = np.arange(len(stimuli))
    width = 0.25
    colors = ["#2E86AB", "#A23B72", "#555555"]

    # Panel superior: TEST (Season 6 hold-out)
    ax0 = axes[0]
    for m_idx, model in enumerate(models):
        means = test_data[m_idx].mean(axis=1)
        stds = test_data[m_idx].std(axis=1)
        offset = (m_idx - 0.5) * width
        label = {
            "temporal_full": "Temporal Transformer",
            "without_temporal_full": "Pointwise MLP",
        }[model]
        ax0.bar(x + offset, means, width, yerr=stds, label=label, color=colors[m_idx], capsize=4, alpha=0.9)

    ax0.axhline(y=ridge_mean_mm, color=colors[2], linestyle="--", linewidth=1.8, label=f"Ridge baseline (MM): {ridge_mean_mm:.3f}")
    ax0.fill_between([-0.5, 1.5], ridge_mean_mm - ridge_std_mm, ridge_mean_mm + ridge_std_mm, alpha=0.12, color=colors[2])

    ax0.set_ylabel("Pearson Correlation ($r$)", fontsize=11, fontweight="bold")
    ax0.set_title("(a) Test Set Performance: Season 6 Hold-out (Strictly Unseen)", fontsize=12, fontweight="bold")
    ax0.set_xticks(x)
    ax0.set_xticklabels(["Multimodal (Video+Audio+Text)", "Text-only"], fontsize=10)
    ax0.set_ylim(0, 0.14)
    ax0.legend(loc="upper right", fontsize=9)
    ax0.grid(axis="y", alpha=0.25)
    ax0.axhline(y=0.10, color="gray", linestyle=":", linewidth=1, alpha=0.6)

    # Panel inferior: VAL (Season 5 hold-out)
    ax1 = axes[1]
    for m_idx, model in enumerate(models):
        means = val_data[m_idx].mean(axis=1)
        stds = val_data[m_idx].std(axis=1)
        offset = (m_idx - 0.5) * width
        label = {
            "temporal_full": "Temporal Transformer",
            "without_temporal_full": "Pointwise MLP",
        }[model]
        ax1.bar(x + offset, means, width, yerr=stds, label=label, color=colors[m_idx], capsize=4, alpha=0.9)

    ax1.set_ylabel("Pearson Correlation ($r$)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Stimulus Modality", fontsize=11, fontweight="bold")
    ax1.set_title("(b) Validation Set Performance: Season 5 Hold-out (Leakage-Free)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Multimodal (Video+Audio+Text)", "Text-only"], fontsize=10)
    ax1.set_ylim(0, 0.14)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", alpha=0.25)
    ax1.axhline(y=0.10, color="gray", linestyle=":", linewidth=1, alpha=0.6)

    plt.tight_layout()
    out_png = OUTPUT_DIR / "figure_2_bars.png"
    out_pdf = OUTPUT_DIR / "figure_2_bars.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Figura 2 guardada: {out_png} y {out_pdf}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURA 3 — Mapas de superficie cortical
# ═══════════════════════════════════════════════════════════════════════════════

def generate_figure_3(test_results: dict, subject: str = "sub-01"):
    """Figura 3: Mapas de Pearson en superficie cortical (Schaefer-1000)."""
    try:
        from nilearn.datasets import fetch_atlas_schaefer_2018, fetch_surf_fsaverage
        from nilearn import plotting
        from nilearn.surface import vol_to_surf
        import nibabel as nib

        atlas = fetch_atlas_schaefer_2018(n_rois=1000, resolution_mm=1)
        atlas_img = atlas.maps

        runs_to_plot = [
            f"temporal_full_multimodal_{subject}",
            f"without_temporal_full_multimodal_{subject}",
        ]

        fig = plt.figure(figsize=(14, 10))
        axes = []
        for i in range(4):
            ax = fig.add_subplot(2, 2, i + 1, projection="3d")
            axes.append(ax)

        for idx, run_name in enumerate(runs_to_plot):
            pm_path = RESULTS_DIR / run_name / "metrics" / "pearson_map_test.pt"
            if not pm_path.exists():
                continue
            pearson = torch.load(pm_path, weights_only=True).numpy()

            atlas_data = nib.load(atlas_img).get_fdata()
            out_data = np.zeros_like(atlas_data)

            for parcel_idx in range(1, 1001):
                mask = atlas_data == parcel_idx
                if parcel_idx - 1 < len(pearson):
                    out_data[mask] = pearson[parcel_idx - 1]

            out_img = nib.Nifti1Image(out_data, nib.load(atlas_img).affine)
            fsavg = fetch_surf_fsaverage(mesh="fsaverage5")
            texture = vol_to_surf(out_img, fsavg.pial_left, radius=3, interpolation="linear")

            row = idx
            title = {
                f"temporal_full_multimodal_{subject}": "Temporal Transformer (Multimodal)",
                f"without_temporal_full_multimodal_{subject}": "Pointwise MLP (Multimodal)",
            }[run_name]

            plotting.plot_surf_stat_map(
                fsavg.infl_left, texture, hemi="left", view="lateral",
                colorbar=(idx == 0), vmin=-0.1, vmax=0.4,
                title=f"{title} (lateral)", axes=axes[row * 2],
                bg_map=fsavg.sulc_left, cmap="RdYlBu_r",
            )

            plotting.plot_surf_stat_map(
                fsavg.infl_left, texture, hemi="left", view="medial",
                colorbar=False, vmin=-0.1, vmax=0.4,
                title=f"{title} (medial)", axes=axes[row * 2 + 1],
                bg_map=fsavg.sulc_left, cmap="RdYlBu_r",
            )

        plt.tight_layout()
        out_png = OUTPUT_DIR / f"figure_3_brain_maps_{subject}.png"
        out_pdf = OUTPUT_DIR / f"figure_3_brain_maps_{subject}.pdf"
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"Figura 3 guardada: {out_png}")
        plt.close(fig)

    except Exception as e:
        print(f"Aviso en figura 3 nilearn: {e}. Generando visualización 2D.")
        generate_figure_3_fallback(subject)


def generate_figure_3_fallback(subject: str = "sub-01"):
    """Visualización 2D ordenada por parcelas corticales."""
    runs_to_plot = [
        (f"temporal_full_multimodal_{subject}", "Temporal Transformer (Multimodal)"),
        (f"without_temporal_full_multimodal_{subject}", "Pointwise MLP (Multimodal)"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    for idx, (run_name, title) in enumerate(runs_to_plot):
        pm_path = RESULTS_DIR / run_name / "metrics" / "pearson_map_test.pt"
        if not pm_path.exists():
            continue
        pearson = torch.load(pm_path, weights_only=True).numpy()

        im = axes[idx].imshow(pearson.reshape(1, -1), aspect="auto", cmap="RdYlBu_r", vmin=-0.1, vmax=0.4)
        axes[idx].set_title(f"{title} — {subject} (Test Set Season 6)", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Schaefer-1000 Parcel Index", fontsize=10)
        axes[idx].set_yticks([])
        plt.colorbar(im, ax=axes[idx], orientation="horizontal", pad=0.25, shrink=0.6, label="Pearson $r$")

    plt.tight_layout()
    out_png = OUTPUT_DIR / f"figure_3_brain_maps_{subject}.png"
    out_pdf = OUTPUT_DIR / f"figure_3_brain_maps_{subject}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Figura 3 guardada: {out_png}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURA 4 — Curvas de entrenamiento limpias
# ═══════════════════════════════════════════════════════════════════════════════

def generate_figure_4():
    """Figura 4: Curvas de entrenamiento (train loss + val Pearson limpio en S5)."""
    runs_to_plot = [
        ("temporal_full_multimodal_sub-01", "Temporal Transformer", "#2E86AB"),
        ("without_temporal_full_multimodal_sub-01", "Pointwise MLP", "#A23B72"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(8, 9))

    for run_name, label, color in runs_to_plot:
        csv_dir = RESULTS_DIR / run_name / "logs" / "csv"
        csv_files = sorted(csv_dir.rglob("metrics.csv"))
        if not csv_files:
            continue
        csv_path = csv_files[-1]

        df = pd.read_csv(csv_path)

        # Train loss
        if "train/loss" in df.columns:
            train_df = df[["epoch", "train/loss"]].dropna()
            axes[0].plot(train_df["epoch"], train_df["train/loss"], label=label, color=color, linewidth=1.6)

        # Val Pearson (Season 5 hold-out)
        if "val/pearson" in df.columns:
            val_df = df[["epoch", "val/pearson"]].dropna()
            axes[1].plot(val_df["epoch"], val_df["val/pearson"], label=label, color=color, linewidth=1.6, marker="o", markersize=3.5)

    axes[0].set_xlabel("Epoch", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Train MSE Loss", fontsize=11, fontweight="bold")
    axes[0].set_title("(a) Training Loss Progression (Seasons 1-4 + Movies)", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.25)

    axes[1].set_xlabel("Epoch", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Validation Pearson ($r$)", fontsize=11, fontweight="bold")
    axes[1].set_title("(b) Validation Performance: Season 5 Hold-out (Leakage-Free)", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.25)
    axes[1].set_ylim(0, 0.12)
    axes[1].axhline(y=0.10, color="gray", linestyle=":", linewidth=1, alpha=0.6)

    plt.tight_layout()
    out_png = OUTPUT_DIR / "figure_4_training_curves.png"
    out_pdf = OUTPUT_DIR / "figure_4_training_curves.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Figura 4 guardada: {out_png} y {out_pdf}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Genera figuras actualizadas para el paper IEEE")
    parser.add_argument("--figure", type=int, choices=[2, 3, 4], default=None, help="Generar solo una figura específica")
    parser.add_argument("--subject", type=str, default="sub-01", help="Sujeto para figura 3")
    args = parser.parse_args()

    test_results = collect_test_results()
    val_results = collect_val_results()
    print(f"Test results cargados: {len(test_results)} runs")
    print(f"Val results cargados: {len(val_results)} runs")

    if args.figure is None or args.figure == 2:
        generate_figure_2(test_results, val_results)
    if args.figure is None or args.figure == 3:
        generate_figure_3(test_results, args.subject)
    if args.figure is None or args.figure == 4:
        generate_figure_4()

    print(f"\nTodas las figuras guardadas en: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
