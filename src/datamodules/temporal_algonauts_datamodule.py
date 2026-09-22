"""
TemporalAlgonautsDataModule — DataModule con ventanas deslizantes y soporte de split episódico limpio.

Para modelos con Temporal Transformer. Cada muestra es una ventana
de W TRs consecutivos construida on-the-fly (baja memoria, ~1 GB).
"""
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L

# pyrefly: ignore [missing-import]
from src.utils.temporal_alignment import HRFAligner


class _SlidingWindowDataset(Dataset):
    """Dataset que genera ventanas deslizantes on-the-fly.

    Almacena solo los tensores base (features, bold) y construye cada
    ventana al indexar. Reduce memoria de ~84 GB a ~1 GB.
    """

    def __init__(self, features: torch.Tensor, bold: torch.Tensor, window_size: int, stride: int = 1):
        self.features = features
        self.bold = bold
        self.window_size = window_size
        self.stride = stride
        self.num_windows = max(0, (features.shape[0] - window_size) // stride + 1)

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        end = start + self.window_size
        return self.features[start:end], self.bold[start:end]


class TemporalAlgonautsDataModule(L.LightningDataModule):
    """
    DataModule temporal para un único sujeto con soporte de split episódico.

    Args:
        features_path: Path a features de train.
        bold_path: Path a BOLD de train.
        val_features_path: Path a features de val (opcional, Season 5).
        val_bold_path: Path a BOLD de val (opcional, Season 5).
        test_features_path: Path a features de test (opcional, Season 6).
        test_bold_path: Path a BOLD de test (opcional, Season 6).
        window_size: TRs por ventana (default 67 ≈ 100s).
        stride: Avance entre ventanas (default 5).
        hrf_delay: Retraso HRF en segundos.
        fmri_tr: TR en segundos.
        val_split: Fracción para validación aleatoria si no se pasa val_features_path.
        batch_size: Batch size.
        normalize_bold: Si True, z-score por parcela.
    """

    def __init__(
        self,
        features_path: str,
        bold_path: str,
        val_features_path: str = None,
        val_bold_path: str = None,
        test_features_path: str = None,
        test_bold_path: str = None,
        window_size: int = 67,
        stride: int = 5,
        hrf_delay: float = 5.0,
        fmri_tr: float = 1.49,
        val_split: float = 0.0,
        batch_size: int = 16,
        normalize_bold: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

    def _prepare_dataset(self, feat_path: str, bold_path: str) -> _SlidingWindowDataset:
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

        return _SlidingWindowDataset(
            features, bold,
            window_size=self.hparams.window_size,
            stride=self.hparams.stride,
        )

    def setup(self, stage: str = None):
        # 1. Dataset de Train
        train_ds = self._prepare_dataset(self.hparams.features_path, self.hparams.bold_path)

        # 2. Dataset de Validación
        if self.hparams.val_features_path and self.hparams.val_bold_path:
            val_ds = self._prepare_dataset(self.hparams.val_features_path, self.hparams.val_bold_path)
            self.train_dataset = train_ds
            self.val_dataset = val_ds
        elif self.hparams.val_split > 0:
            total = len(train_ds)
            val_size = int(total * self.hparams.val_split)
            train_size = total - val_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                train_ds,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )
        else:
            self.train_dataset = train_ds
            self.val_dataset = train_ds

        # 3. Dataset de Test
        if self.hparams.test_features_path and self.hparams.test_bold_path:
            self.test_dataset = self._prepare_dataset(self.hparams.test_features_path, self.hparams.test_bold_path)
        else:
            self.test_dataset = self.val_dataset

        print(
            f"TemporalDataModule: train={len(self.train_dataset)} windows, "
            f"val={len(self.val_dataset)} windows, test={len(self.test_dataset)} windows"
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=0)
