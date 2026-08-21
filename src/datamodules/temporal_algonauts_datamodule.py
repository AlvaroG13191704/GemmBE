"""
TemporalAlgonautsDataModule — DataModule con ventanas deslizantes.

Para modelos con Temporal Transformer. Cada muestra es una ventana
de W TRs consecutivos.

OPTIMIZACION DE MEMORIA: en lugar de materializar todas las ventanas
en RAM (83+ GB), se construyen on-the-fly en __getitem__. Esto reduce
el uso de memoria a ~1 GB (solo los tensores base alineados).
"""
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
        self.num_windows = (features.shape[0] - window_size) // stride + 1

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        end = start + self.window_size
        return self.features[start:end], self.bold[start:end]


class TemporalAlgonautsDataModule(L.LightningDataModule):
    """
    DataModule temporal para un unico sujeto.

    Args:
        features_path: Path a real_stimulus_features.pt.
        bold_path: Path a sub-XX.pt.
        window_size: TRs por ventana (default 67 ~ 100s).
        stride: Avance entre ventanas (default 1 = maximo solapamiento).
        hrf_delay: Retraso HRF en segundos.
        fmri_tr: TR en segundos.
        val_split: Fraccion para validacion.
        batch_size: Batch size.
        normalize_bold: Si True, z-score por parcela.
    """

    def __init__(
        self,
        features_path: str,
        bold_path: str,
        window_size: int = 67,
        stride: int = 5,
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

        # Dataset con ventanas on-the-fly (baja memoria)
        full_dataset = _SlidingWindowDataset(
            features, bold,
            window_size=self.hparams.window_size,
            stride=self.hparams.stride,
        )
        total = len(full_dataset)
        val_size = int(total * self.hparams.val_split)
        train_size = total - val_size

        if val_size > 0:
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )
        else:
            self.train_dataset = full_dataset
            self.val_dataset = full_dataset

        # Memoria estimada: solo tensores base
        feat_mb = features.numel() * features.element_size() / (1024 ** 2)
        bold_mb = bold.numel() * bold.element_size() / (1024 ** 2)
        print(
            f"TemporalDataModule: train={train_size}, val={val_size}, "
            f"windows={total}, mem_base={feat_mb+bold_mb:.1f}MB"
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=0)
