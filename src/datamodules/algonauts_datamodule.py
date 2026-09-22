"""
AlgonautsDataModule — LightningDataModule para features + fMRI pointwise (sin ventanas temporales).

Carga tensores pre-extraídos, aplica alineación HRF, y genera DataLoaders de train/val/test
con soporte de split episódico limpio.
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import lightning as L

# pyrefly: ignore [missing-import]
from src.utils.temporal_alignment import HRFAligner


class AlgonautsDataModule(L.LightningDataModule):
    """
    DataModule pointwise para un único sujeto de Algonauts 2025.

    Args:
        features_path: Path a features de train (T, 1536).
        bold_path: Path a BOLD de train (T, 1000).
        val_features_path: Path a features de val (opcional).
        val_bold_path: Path a BOLD de val (opcional).
        test_features_path: Path a features de test (opcional).
        test_bold_path: Path a BOLD de test (opcional).
        hrf_delay: Retraso hemodinámico en segundos (5.0 o 0.0).
        fmri_tr: TR en segundos (1.49).
        val_split: Fracción para validación si no se pasa val_features_path.
        batch_size: Batch size para entrenamiento.
        normalize_bold: Si True, aplica z-score por parcela.
    """

    def __init__(
        self,
        features_path: str,
        bold_path: str,
        val_features_path: str = None,
        val_bold_path: str = None,
        test_features_path: str = None,
        test_bold_path: str = None,
        hrf_delay: float = 5.0,
        fmri_tr: float = 1.49,
        val_split: float = 0.0,
        batch_size: int = 64,
        normalize_bold: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

    def _prepare_dataset(self, feat_path: str, bold_path: str) -> TensorDataset:
        features = torch.load(feat_path, weights_only=True)
        bold = torch.load(bold_path, weights_only=True)

        aligner = HRFAligner(
            hrf_delay_seconds=self.hparams.hrf_delay,
            fmri_tr_seconds=self.hparams.fmri_tr,
        )
        features, bold = aligner.align_stimulus_to_fmri(features, bold)

        if self.hparams.normalize_bold:
            mean = bold.mean(dim=0, keepdim=True)
            std = bold.std(dim=0, keepdim=True).clamp(min=1e-8)
            bold = (bold - mean) / std

        return TensorDataset(features, bold)

    def setup(self, stage: str = None):
        train_ds = self._prepare_dataset(self.hparams.features_path, self.hparams.bold_path)

        if self.hparams.val_features_path and self.hparams.val_bold_path:
            val_ds = self._prepare_dataset(self.hparams.val_features_path, self.hparams.val_bold_path)
            self.train_dataset = train_ds
            self.val_dataset = val_ds
        elif self.hparams.val_split > 0:
            total = len(train_ds)
            val_size = int(total * self.hparams.val_split)
            train_size = total - val_size
            self.train_dataset, self.val_dataset = random_split(
                train_ds,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )
        else:
            self.train_dataset = train_ds
            self.val_dataset = train_ds

        if self.hparams.test_features_path and self.hparams.test_bold_path:
            self.test_dataset = self._prepare_dataset(self.hparams.test_features_path, self.hparams.test_bold_path)
        else:
            self.test_dataset = self.val_dataset

        print(
            f"AlgonautsDataModule: train={len(self.train_dataset)} samples, "
            f"val={len(self.val_dataset)} samples, test={len(self.test_dataset)} samples"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=0,
        )
