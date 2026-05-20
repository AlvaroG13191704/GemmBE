"""
================================================================================
train.py — Entrenamiento masivo con PyTorch Lightning
================================================================================

Grid para paper (8 combinaciones):
  • temporal_full:       multimodal + textonly × sub-01 + sub-02 = 4 runs
  • without_temporal:    multimodal + textonly × sub-01 + sub-02 = 4 runs

Preguntas que responde esta grid:
  1. ¿Cuánto mejora el Temporal Transformer? → temporal_full vs without_temporal
  2. ¿Multimodal > text-only? → ambos estímulos en temporal_full y without_temporal

Uso:
    python train.py
    python train.py --epochs 100 --batch_size 64
    python train.py --dry_run

Cada run guarda automáticamente:
  • checkpoints/     — pesos del modelo (.ckpt)
  • metrics/         — pearson_map_val.pt, pearson_map_test.pt, test_results.json
  • logs/            — TensorBoard + CSV con historial completo
================================================================================
"""

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

# pyrefly: ignore [missing-import]
from src.models import (
    TemporalFullModel,
    WithoutTemporalFullModel,
)
# pyrefly: ignore [missing-import]
from src.datamodules import (
    AlgonautsDataModule,
    TemporalAlgonautsDataModule,
)
# pyrefly: ignore [missing-import]
from src.callbacks import MetricsCallback


ALGONAUTS_SUBJECTS = ["sub-01", "sub-02"]

MODEL_REGISTRY = {
    "temporal_full": {
        "model_cls": TemporalFullModel,
        "datamodule_cls": TemporalAlgonautsDataModule,
        "hrf": 5.0,
        "temporal": True,
        "stimuli": ["multimodal", "textonly"],
    },
    "without_temporal_full": {
        "model_cls": WithoutTemporalFullModel,
        "datamodule_cls": AlgonautsDataModule,
        "hrf": 5.0,
        "temporal": False,
        "stimuli": ["multimodal", "textonly"],
    },
    "no_hrf": {
        "model_cls": TemporalFullModel,
        "datamodule_cls": TemporalAlgonautsDataModule,
        "hrf": 0.0,
        "temporal": True,
        "stimuli": ["multimodal"],
    },
}

STIMULUS_DIRS = {
    "multimodal": "data/features",
    "textonly": "data/features_text_only",
}


def run_experiment(
    model_key: str,
    stimulus_key: str,
    subject_id: str,
    epochs: int,
    batch_size: int,
    lr: float,
    fmri_dir: str,
    root_results: str,
    stride: int = 5,
):
    """Ejecuta un único experimento de la grid."""
    cfg = MODEL_REGISTRY[model_key]
    stim_dir = Path(STIMULUS_DIRS[stimulus_key])
    features_path = stim_dir / "real_stimulus_features.pt"

    if not features_path.exists():
        print(f" Features no encontrados: {features_path}. Saltando.")
        return

    run_name = f"{model_key}_{stimulus_key}_{subject_id}"
    save_dir = Path(root_results) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"{run_name}")
    print(f"Features: {features_path}")
    print(f"HRF:      {cfg['hrf']}s")
    print(f"{'='*60}")

    # ─── DataModule ──────────────────────────────────────────────────────────
    fmri_path = Path(fmri_dir) / f"{subject_id}.pt"
    if not fmri_path.exists():
        print(f" fMRI no encontrado: {fmri_path}. Saltando.")
        return
    dm_kwargs = {
        "features_path": str(features_path),
        "bold_path": str(fmri_path),
        "hrf_delay": cfg["hrf"],
        "fmri_tr": 1.49,
        "val_split": 0.1,
        "batch_size": batch_size if not cfg["temporal"] else batch_size // 4,
        "normalize_bold": True,
    }
    if cfg["temporal"]:
        dm_kwargs["stride"] = stride
    dm = cfg["datamodule_cls"](**dm_kwargs)

    # ─── Model ───────────────────────────────────────────────────────────────
    model_kwargs = {
        "stimulus_type": stimulus_key,
        "subject_id": subject_id,
        "num_vertices": 1000,
        "lr": lr,
        "weight_decay": 1e-5,
        "max_epochs": epochs,
    }
    if cfg["temporal"]:
        model_kwargs["window_size"] = 67
    model = cfg["model_cls"](**model_kwargs)

    # ─── Callbacks ───────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath=save_dir / "checkpoints",
        filename=f"{run_name}_epoch={{epoch:03d}}_pearson={{val/pearson:.4f}}",
        monitor="val/pearson",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/pearson",
        patience=20,
        mode="max",
        verbose=False,
    )
    metrics_cb = MetricsCallback(save_dir=save_dir / "metrics")

    # ─── Loggers ─────────────────────────────────────────────────────────────
    tb_logger = TensorBoardLogger(save_dir=save_dir / "logs", name="tb")
    csv_logger = CSVLogger(save_dir=save_dir / "logs", name="csv")

    # ─── Trainer ─────────────────────────────────────────────────────────────
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        logger=[tb_logger, csv_logger],
        callbacks=[checkpoint_cb, early_stop_cb, metrics_cb],
        enable_progress_bar=True,
        default_root_dir=str(save_dir),
        check_val_every_n_epoch=5,
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")

    print(f"Completado: {run_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento masivo de brain encoding con PyTorch Lightning"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fmri_dir", type=str, default="data/subjects_fmri_filtered")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--stimuli", nargs="+", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Stride para ventanas temporales (solo modelos temporales). 1=maximo solapamiento, 5=5x mas rapido.",
    )

    args = parser.parse_args()

    subjects = args.subjects or ALGONAUTS_SUBJECTS
    models = args.models or list(MODEL_REGISTRY.keys())
    stimuli = args.stimuli or list(STIMULUS_DIRS.keys())

    invalid_models = [m for m in models if m not in MODEL_REGISTRY]
    if invalid_models:
        raise ValueError(f"Modelos inválidos: {invalid_models}. Opciones: {list(MODEL_REGISTRY.keys())}")
    invalid_stimuli = [s for s in stimuli if s not in STIMULUS_DIRS]
    if invalid_stimuli:
        raise ValueError(f"Estímulos inválidos: {invalid_stimuli}. Opciones: {list(STIMULUS_DIRS.keys())}")

    # Construir grid respetando qué estímulos puede usar cada modelo
    experiments = []
    for m in models:
        allowed_stimuli = set(MODEL_REGISTRY[m]["stimuli"])
        for s in stimuli:
            if s not in allowed_stimuli:
                continue
            for sub in subjects:
                experiments.append((m, s, sub))

    print(f"\nGrid de experimentos: {len(experiments)} combinaciones")
    print(f"Modelos:  {models}")
    print(f"Estímulos: {stimuli}")
    print(f"Sujetos:  {subjects}")
    print(f"Épocas:   {args.epochs}")
    print(f"Batch:    {args.batch_size}")

    if args.dry_run:
        print("\n📋 Dry run — combinaciones planificadas:")
        for m, s, sub in experiments:
            print(f"   {m}_{s}_{sub}")
        return

    for i, (m, s, sub) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] ", end="")
        try:
            run_experiment(
                model_key=m,
                stimulus_key=s,
                subject_id=sub,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                fmri_dir=args.fmri_dir,
                root_results=args.results_dir,
                stride=args.stride,
            )
        except Exception as e:
            print(f"\n Error en {m}_{s}_{sub}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print("Todos los experimentos completados.")
    print(f"Resultados en: {args.results_dir}/")
    print(f"Para ver logs: tensorboard --logdir={args.results_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
