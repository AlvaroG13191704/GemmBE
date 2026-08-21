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
    TriBEStyleModel,
)
# pyrefly: ignore [missing-import]
from src.datamodules import (
    AlgonautsDataModule,
    TemporalAlgonautsDataModule,
    TriBEDataModule,
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
    # ─── Modelos TriBE-Style (v3) ────────────────────────────────────────────
    "tribe_full": {
        "model_cls": TriBEStyleModel,
        "datamodule_cls": TriBEDataModule,
        "hrf": 5.0,
        "temporal": True,
        "stimuli": ["multimodal"],
        "ablation": "full",
    },
    "tribe_vit_only": {
        "model_cls": TriBEStyleModel,
        "datamodule_cls": TriBEDataModule,
        "hrf": 5.0,
        "temporal": True,
        "stimuli": ["multimodal"],
        "ablation": "vit_only",
    },
    "tribe_conformer_only": {
        "model_cls": TriBEStyleModel,
        "datamodule_cls": TriBEDataModule,
        "hrf": 5.0,
        "temporal": True,
        "stimuli": ["multimodal"],
        "ablation": "conformer_only",
    },
    "tribe_text_only": {
        "model_cls": TriBEStyleModel,
        "datamodule_cls": TriBEDataModule,
        "hrf": 5.0,
        "temporal": True,
        "stimuli": ["multimodal"],
        "ablation": "text_only",
    },
    "tribe_vit_conformer": {
        "model_cls": TriBEStyleModel,
        "datamodule_cls": TriBEDataModule,
        "hrf": 5.0,
        "temporal": True,
        "stimuli": ["multimodal"],
        "ablation": "vit_conformer",
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
    root_results: str,
    stride: int = 5,
    split_dir: str = "data/train_test_split",
):
    """Ejecuta un único experimento de la grid con train/test split estricto."""
    cfg = MODEL_REGISTRY[model_key]
    is_tribe = "tribe" in model_key

    # Auto-ajustar directorio de split por defecto para modelos TriBE
    if is_tribe and split_dir == "data/train_test_split":
        split_dir = "data/train_test_split_v3"

    split_path = Path(split_dir)
    
    # ─── Verificación de archivos requeridos ──────────────────────────────────
    if is_tribe:
        required_paths = [
            split_path / "vit_train.pt",
            split_path / "vit_test.pt",
            split_path / "conformer_train.pt",
            split_path / "conformer_test.pt",
            split_path / "text_train.pt",
            split_path / "text_test.pt",
            split_path / f"bold_train_{subject_id}.pt",
            split_path / f"bold_test_{subject_id}.pt",
        ]
    else:
        required_paths = [
            split_path / "features_train.pt",
            split_path / "features_test.pt",
            split_path / f"bold_train_{subject_id}.pt",
            split_path / f"bold_test_{subject_id}.pt",
        ]

    for p in required_paths:
        if not p.exists():
            print(f" Archivo no encontrado: {p}")
            if is_tribe:
                print("   Ejecuta primero: python -m src.prepare_train_test_split_v3")
            else:
                print("   Ejecuta primero: uv run python -m src.prepare_train_test_split")
            return

    run_name = f"{model_key}_{stimulus_key}_{subject_id}"
    save_dir = Path(root_results) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"{run_name}")
    print(f"Split dir:     {split_path}")
    print(f"HRF:           {cfg['hrf']}s")
    print(f"{'='*60}")

    # ─── DataModule ──────────────────────────────────────────────────────────
    if is_tribe:
        dm_kwargs = {
            "vit_path": str(split_path / "vit_train.pt"),
            "conformer_path": str(split_path / "conformer_train.pt"),
            "text_path": str(split_path / "text_train.pt"),
            "bold_path": str(split_path / f"bold_train_{subject_id}.pt"),
            "window_size": 67,
            "stride": stride,
            "hrf_delay": cfg["hrf"],
            "fmri_tr": 1.49,
            "val_split": 0.1,
            "batch_size": batch_size // 4,
            "normalize_bold": False,   # Ya normalizado en split
            "normalize_feats": True,
        }
    else:
        features_train = split_path / "features_train.pt"
        dm_kwargs = {
            "features_path": str(features_train),
            "bold_path": str(split_path / f"bold_train_{subject_id}.pt"),
            "hrf_delay": cfg["hrf"],
            "fmri_tr": 1.49,
            "val_split": 0.1,
            "batch_size": batch_size if not cfg["temporal"] else batch_size // 4,
            "normalize_bold": False,  # Ya normalizado en split
        }
        if cfg["temporal"]:
            dm_kwargs["stride"] = stride
            
    dm = cfg["datamodule_cls"](**dm_kwargs)

    # ─── Model ───────────────────────────────────────────────────────────────
    if is_tribe:
        model_kwargs = {
            "window_size": 67,
            "ablation": cfg["ablation"],
            "num_subjects": 2,
            "subject_id": subject_id,
            "num_vertices": 1000,
            "lr": lr,
            "weight_decay": 1e-5,
            "max_epochs": epochs,
        }
    else:
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

    # ─── Test en Season 6 (hold-out estricto) ─────────────────────────────
    print(f"\n{'='*60}")
    print("Evaluando en TEST SET (Season 6 hold-out)")
    print(f"{'='*60}")

    if is_tribe:
        test_dm_kwargs = {
            "vit_path": str(split_path / "vit_test.pt"),
            "conformer_path": str(split_path / "conformer_test.pt"),
            "text_path": str(split_path / "text_test.pt"),
            "bold_path": str(split_path / f"bold_test_{subject_id}.pt"),
            "window_size": 67,
            "stride": stride,
            "hrf_delay": cfg["hrf"],
            "fmri_tr": 1.49,
            "val_split": 0.0,  # Sin validación en test
            "batch_size": batch_size // 4,
            "normalize_bold": False,
            "normalize_feats": True,
        }
    else:
        features_test = split_path / "features_test.pt"
        test_dm_kwargs = {
            "features_path": str(features_test),
            "bold_path": str(split_path / f"bold_test_{subject_id}.pt"),
            "hrf_delay": cfg["hrf"],
            "fmri_tr": 1.49,
            "val_split": 0.0,
            "batch_size": batch_size if not cfg["temporal"] else batch_size // 4,
            "normalize_bold": False,
        }
        if cfg["temporal"]:
            test_dm_kwargs["stride"] = stride

    test_dm = cfg["datamodule_cls"](**test_dm_kwargs)

    # Load best checkpoint for test
    # Lightning saves best checkpoint in a subdirectory; find it
    ckpt_dir = save_dir / "checkpoints"
    ckpt_files = sorted(ckpt_dir.glob("*.ckpt"))
    if ckpt_files:
        ckpt_path = str(ckpt_files[-1])  # Last one should be best (sorted by epoch)
    else:
        # Fallback: search in subdirectories
        ckpt_files = sorted(ckpt_dir.rglob("*.ckpt"))
        if ckpt_files:
            # Find the one with highest pearson in filename
            best_ckpt = None
            best_pearson = -1.0
            for cf in ckpt_files:
                # Parse pearson value from filename like "pearson=0.6850.ckpt"
                if "pearson=" in cf.name:
                    try:
                        p_str = cf.name.split("pearson=")[1].replace(".ckpt", "")
                        p_val = float(p_str)
                        if p_val > best_pearson:
                            best_pearson = p_val
                            best_ckpt = cf
                    except (ValueError, IndexError):
                        continue
            ckpt_path = str(best_ckpt) if best_ckpt else None
        else:
            ckpt_path = None
    
    print(f"  Test con checkpoint: {ckpt_path}")
    trainer.test(model, datamodule=test_dm, ckpt_path=ckpt_path)

    print(f"Completado: {run_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento masivo de brain encoding con PyTorch Lightning"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--split_dir", type=str, default="data/train_test_split")
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--stimuli", nargs="+", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Stride temporal (default 5, corta steps/epoch 5x)",
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
                root_results=args.results_dir,
                stride=args.stride,
                split_dir=args.split_dir,
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
