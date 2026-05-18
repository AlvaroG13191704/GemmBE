"""
AlgonautsDataModule — LightningDataModule para features + fMRI.

Carga tensores pre-extraídos, aplica alineación HRF, normaliza BOLD,
y genera DataLoaders de train/val.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
import lightning as L

# pyrefly: ignore [missing-import]
from src.utils.temporal_alignment import HRFAligner


class AlgonautsDataModule(L.LightningDataModule):
    """
    DataModule para un único sujeto de Algonauts 2025.

    Args:
        features_path: Path a real_stimulus_features.pt (T, 1536).
        bold_path: Path a sub-XX.pt (T, 1000).
        hrf_delay: Retraso hemodinámico en segundos (5.0 o 0.0).
        fmri_tr: TR en segundos (1.49).
        val_split: Fracción para validación (0.1 = 10%).
        batch_size: Batch size para entrenamiento.
        normalize_bold: Si True, aplica z-score por parcela.
    """

    def __init__(
        self,
        features_path: str,
        bold_path: str,
        hrf_delay: float = 5.0,
        fmri_tr: float = 1.49,
        val_split: float = 0.1,
        batch_size: int = 64,
        normalize_bold: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.features_path = Path(features_path)
        self.bold_path = Path(bold_path)

    def setup(self, stage: str = None):
        features = torch.load(self.features_path, weights_only=True)
        bold = torch.load(self.bold_path, weights_only=True)

        # Alineación HRF
        aligner = HRFAligner(
            hrf_delay_seconds=self.hparams.hrf_delay,
            fmri_tr_seconds=self.hparams.fmri_tr,
        )
        features, bold = aligner.align_stimulus_to_fmri(features, bold)

        # Normalización z-score por parcela
        if self.hparams.normalize_bold:
            mean = bold.mean(dim=0, keepdim=True)
            std = bold.std(dim=0, keepdim=True).clamp(min=1e-8)
            bold = (bold - mean) / std

        # Dataset unificado
        dataset = torch.utils.data.TensorDataset(features, bold)
        total = len(dataset)
        val_size = int(total * self.hparams.val_split)
        train_size = total - val_size

        self.train_dataset, self.val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        print(f"📊 DataModule: train={train_size}, val={val_size}, total_features={features.shape}")

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
        # Reusa val como test para consistencia
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=0,
        )
