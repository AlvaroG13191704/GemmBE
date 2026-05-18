"""
TemporalAlgonautsDataModule — DataModule con ventanas deslizantes.

Para modelos con Temporal Transformer. Cada muestra es una ventana
de W TRs consecutivos.
"""
import torch
from torch.utils.data import DataLoader
import lightning as L

# pyrefly: ignore [missing-import]
from src.utils.temporal_alignment import HRFAligner


class TemporalAlgonautsDataModule(L.LightningDataModule):
    """
    DataModule temporal para un único sujeto.

    Args:
        features_path: Path a real_stimulus_features.pt.
        bold_path: Path a sub-XX.pt.
        window_size: TRs por ventana (default 67 ≈ 100s).
        stride: Avance entre ventanas (default 1 = máximo solapamiento).
        hrf_delay: Retraso HRF en segundos.
        fmri_tr: TR en segundos.
        val_split: Fracción para validación.
        batch_size: Batch size.
        normalize_bold: Si True, z-score por parcela.
    """

    def __init__(
        self,
        features_path: str,
        bold_path: str,
        window_size: int = 67,
        stride: int = 1,
        hrf_delay: float = 5.0,
        fmri_tr: float = 1.49,
        val_split: float = 0.1,
        batch_size: int = 16,
        normalize_bold: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str = None):
        features = torch.load(self.hparams.features_path, weights_only=True)
        bold = torch.load(self.hparams.bold_path, weights_only=True)

        aligner = HRFAligner(
            hrf_delay_seconds=self.hparams.hrf_delay,
            fmri_tr_seconds=self.hparams.fmri_tr,
        )
        features, bold = aligner.align_stimulus_to_fmri(features, bold)

        if self.hparams.normalize_bold:
            mean = bold.mean(dim=0, keepdim=True)
            std = bold.std(dim=0, keepdim=True).clamp(min=1e-8)
            bold = (bold - mean) / std

        # Crear ventanas deslizantes
        W = self.hparams.window_size
        stride = self.hparams.stride
        T = features.shape[0]

        feat_windows, bold_windows = [], []
        for start in range(0, T - W + 1, stride):
            end = start + W
            feat_windows.append(features[start:end])
            bold_windows.append(bold[start:end])

        feat_windows = torch.stack(feat_windows)  # (N, W, 1536)
        bold_windows = torch.stack(bold_windows)  # (N, W, 1000)

        dataset = torch.utils.data.TensorDataset(feat_windows, bold_windows)
        total = len(dataset)
        val_size = int(total * self.hparams.val_split)
        train_size = total - val_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        print(f"📊 TemporalDataModule: train={train_size}, val={val_size}, windows={total}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=0)
