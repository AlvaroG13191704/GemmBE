"""
Generador de figuras para paper IEEE — MicroTRIBE-Gemma.

Figuras:
  2. Barras comparativas de Pearson medio (TEST = Season 6 hold-out)
  3. Mapas de superficie cortical (Pearson por parcela)
  4. Curvas de entrenamiento (train loss + val Pearson)

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
    """Recopila val_pearson de los CSVs."""
    val_results = {}
    for csv_path in sorted(RESULTS_DIR.glob("*/logs/csv/version_0/metrics.csv")):
        run_name = csv_path.parent.parent.parent.parent.name
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r['val/pearson'] and r['val/pearson'].strip()]
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
# FIGURA 2 — Barras comparativas (TEST = Season 6 hold-out honesto)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_figure_2(test_results: dict, val_results: dict):
    """Figura 2: Barras comparativas de Pearson medio TEST (Season 6 hold-out)."""
    models = ["temporal_full", "without_temporal_full"]
    stimuli = ["multimodal", "textonly"]
    subjects = ["sub-01", "sub-02"]

    # TEST data: modelo x estímulo x sujeto
    test_data = np.zeros((len(models), len(stimuli), len(subjects)))
    for run_name, metrics in test_results.items():
        model, stimulus, subject = parse_run_name(run_name)
        if model not in models or stimulus not in stimuli:
            continue
        m_idx = models.index(model)
        s_idx = stimuli.index(stimulus)
        sub_idx = subjects.index(subject)
        test_data[m_idx, s_idx, sub_idx] = metrics["test_pearson_mean"]

    # VAL data (para comparación transparente)
    val_data = np.zeros((len(models), len(stimuli), len(subjects)))
    for run_name, val_pearson in val_results.items():
        model, stimulus, subject = parse_run_name(run_name)
        if model not in models or stimulus not in stimuli:
            continue
        m_idx = models.index(model)
        s_idx = stimuli.index(stimulus)
        sub_idx = subjects.index(subject)
        val_data[m_idx, s_idx, sub_idx] = val_pearson

    # Ridge baseline
    ridge_test = []
    for sub in subjects:
        ridge_file = RESULTS_DIR / f"ridge_baseline/ridge_{sub}/test_results.json"
        if ridge_file.exists():
            with open(ridge_file) as f:
                data = json.load(f)
            ridge_test.append(data["test_pearson"])
    ridge_mean = np.mean(ridge_test) if ridge_test else 0.0
    ridge_std = np.std(ridge_test) if ridge_test else 0.0

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    x = np.arange(len(stimuli))
    width = 0.25
    colors = ["#2E86AB", "#A23B72", "#888888"]

    # Panel superior: TEST (Season 6 hold-out)
    ax = axes[0]
    for m_idx, model in enumerate(models):
        means = test_data[m_idx].mean(axis=1)
        stds = test_data[m_idx].std(axis=1)
        offset = (m_idx - 1) * width
        label = {
            "temporal_full": "Transformer",
            "without_temporal_full": "Linear (MLP)",
        }[model]
        ax.bar(x + offset, means, width, yerr=stds, label=label, color=colors[m_idx], capsize=4)

    # Ridge como línea horizontal
    ax.axhline(y=ridge_mean, color=colors[2], linestyle="--", linewidth=2, label=f"Ridge baseline: {ridge_mean:.3f}")
    ax.fill_between([-0.5, 1.5], ridge_mean - ridge_std, ridge_mean + ridge_std, alpha=0.15, color=colors[2])

    ax.set_ylabel("Pearson Correlation", fontsize=12)
    ax.set_xlabel("Stimulus Modality", fontsize=12)
    ax.set_title("(a) Test Set: Season 6 Hold-out", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Multimodal", "Text-only"], fontsize=11)
    ax.set_ylim(0, 0.15)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0.15, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(1.45, 0.152, "r=0.15", fontsize=8, color="gray", ha="right")

    # Panel inferior: VAL (random 10% de S1-S5, leakage warning)
    ax = axes[1]
    for m_idx, model in enumerate(models):
        means = val_data[m_idx].mean(axis=1)
        stds = val_data[m_idx].std(axis=1)
        offset = (m_idx - 1) * width
        label = {
            "temporal_full": "Transformer",
            "without_temporal_full": "Linear (MLP)",
        }[model]
        ax.bar(x + offset, means, width, yerr=stds, label=label, color=colors[m_idx], capsize=4)

    ax.set_ylabel("Pearson Correlation", fontsize=12)
    ax.set_xlabel("Stimulus Modality", fontsize=12)
    ax.set_title("(b) Validation Set (S1-S5 subset, temporal leakage)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Multimodal", "Text-only"], fontsize=11)
    ax.set_ylim(0, 0.85)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0.15, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(1.45, 0.152, "r=0.15", fontsize=8, color="gray", ha="right")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "figure_2_bars.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "figure_2_bars.pdf", bbox_inches="tight")
    print(f"Figura 2 guardada: {out_path}")
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
            if run_name not in test_results:
                continue

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
                f"temporal_full_multimodal_{subject}": "Transformer (with HRF) - Multimodal",
                f"without_temporal_full_multimodal_{subject}": "Linear (without Transformer) - Multimodal",
            }[run_name]

            plotting.plot_surf_stat_map(
                fsavg.infl_left, texture, hemi="left", view="lateral",
                colorbar=(idx == 0), vmin=-0.2, vmax=0.9,
                title=f"{title} (lateral)", axes=axes[row * 2],
                bg_map=fsavg.sulc_left, cmap="RdYlBu_r",
            )

            plotting.plot_surf_stat_map(
                fsavg.infl_left, texture, hemi="left", view="medial",
                colorbar=False, vmin=-0.2, vmax=0.9,
                title=f"{title} (medial)", axes=axes[row * 2 + 1],
                bg_map=fsavg.sulc_left, cmap="RdYlBu_r",
            )

        plt.tight_layout()
        out_path = OUTPUT_DIR / f"figure_3_brain_maps_{subject}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        fig.savefig(OUTPUT_DIR / f"figure_3_brain_maps_{subject}.pdf", bbox_inches="tight")
        print(f"Figura 3 guardada: {out_path}")
        plt.close(fig)

    except Exception as e:
        print(f"Error generando figura 3: {e}")
        generate_figure_3_fallback(test_results, subject)


def generate_figure_3_fallback(test_results: dict, subject: str):
    """Fallback si nilearn surface plotting falla."""
    runs_to_plot = [
        (f"temporal_full_multimodal_{subject}", "Transformer (with HRF)"),
        (f"without_temporal_full_multimodal_{subject}", "Linear (without Transformer)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for idx, (run_name, title) in enumerate(runs_to_plot):
        if run_name not in test_results:
            continue
        pm_path = RESULTS_DIR / run_name / "metrics" / "pearson_map_test.pt"
        if not pm_path.exists():
            continue
        pearson = torch.load(pm_path, weights_only=True).numpy()

        im = axes[idx].imshow(pearson.reshape(1, -1), aspect="auto", cmap="RdYlBu_r", vmin=-0.2, vmax=0.9)
        axes[idx].set_title(title, fontsize=11)
        axes[idx].set_xlabel("Parcel (1-1000)", fontsize=10)
        axes[idx].set_yticks([])
        plt.colorbar(im, ax=axes[idx], label="Pearson r")

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"figure_3_fallback_{subject}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Figura 3 (fallback) guardada: {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURA 4 — Curvas de entrenamiento
# ═══════════════════════════════════════════════════════════════════════════════

def generate_figure_4():
    """Figura 4: Curvas de entrenamiento (train loss + val Pearson)."""
    runs_to_plot = [
        ("temporal_full_multimodal_sub-01", "Transformer (with HRF)", "#2E86AB"),
        ("without_temporal_full_multimodal_sub-01", "Linear (without Transformer)", "#A23B72"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    for run_name, label, color in runs_to_plot:
        csv_path = RESULTS_DIR / run_name / "logs" / "csv" / "version_0" / "metrics.csv"
        if not csv_path.exists():
            versions = list((RESULTS_DIR / run_name / "logs" / "csv").glob("version_*"))
            if versions:
                csv_path = versions[0] / "metrics.csv"
            else:
                continue

        df = pd.read_csv(csv_path)

        # Train loss
        train_loss = df["train/loss"].dropna()
        train_epochs = np.arange(len(train_loss))
        axes[0].plot(train_epochs, train_loss, label=label, color=color, linewidth=1.5)

        # Val Pearson
        val_pearson = df["val/pearson"].dropna()
        val_epochs = np.arange(4, 4 + len(val_pearson) * 5, 5)[:len(val_pearson)]
        axes[1].plot(val_epochs, val_pearson, label=label, color=color, linewidth=1.5, marker="o", markersize=4)

    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Train MSE Loss", fontsize=11)
    axes[0].set_title("(a) Training Loss", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Validation Pearson r", fontsize=11)
    axes[1].set_title("(b) Validation Performance (S1-S5)", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 0.85)

    # Añadir anotación de leakage
    axes[1].text(0.98, 0.98, "Note: Val set from same episodes as train\n(temporal window overlap)",
                 transform=axes[1].transAxes, fontsize=8, verticalalignment='top',
                 horizontalalignment='right', color='red', alpha=0.7,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Línea de umbral
    axes[1].axhline(y=0.15, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "figure_4_training_curves.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "figure_4_training_curves.pdf", bbox_inches="tight")
    print(f"Figura 4 guardada: {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Genera figuras para el paper IEEE")
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
