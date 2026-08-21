"""
tribe_datamodule.py — DataModule para el modelo TriBEStyleModel.

Carga los 3 tensores separados producidos por extract_features_v3.py:
  • real_vit_features.pt        (N_TRs, D_vit)
  • real_conformer_features.pt  (N_TRs, D_conformer)
  • real_text_features.pt       (N_TRs, 1536)
  • bold (sub-XX.pt)            (N_TRs, 1000)

Cada muestra del dataset es una tupla:
  (vit_window, conformer_window, text_window, bold_window)
donde cada ventana tiene forma (W, D_modalidad).

El HRF alignment y la normalización z-score se aplican igual que en v2,
pero de forma independiente para cada modalidad (evita que la dimensión
mayor domine al normalizar globalmente).
"""

import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L

# pyrefly: ignore [missing-import]
from src.utils.temporal_alignment import HRFAligner


class _TriBESlidingWindowDataset(Dataset):
    """
    Dataset con ventanas deslizantes para las 3 modalidades + BOLD.

    Almacena solo los tensores base y construye ventanas on-the-fly
    para mantener el uso de RAM a ~3-5 MB (en vez de ~80 GB).
    """

    def __init__(
        self,
        vit:       torch.Tensor,   # (N, D_vit)
        conformer: torch.Tensor,   # (N, D_conf)
        text:      torch.Tensor,   # (N, D_text)
        bold:      torch.Tensor,   # (N, 1000)
        window_size: int,
        stride: int = 1,
    ):
        assert vit.shape[0] == conformer.shape[0] == text.shape[0] == bold.shape[0], (
            f"Inconsistencia de TRs: vit={vit.shape[0]}, "
            f"conformer={conformer.shape[0]}, "
            f"text={text.shape[0]}, bold={bold.shape[0]}"
        )
        self.vit       = vit
        self.conformer = conformer
        self.text      = text
        self.bold      = bold
        self.window_size   = window_size
        self.stride        = stride
        self.num_windows   = (vit.shape[0] - window_size) // stride + 1

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        s = idx * self.stride
        e = s + self.window_size
        return (
            self.vit[s:e],
            self.conformer[s:e],
            self.text[s:e],
            self.bold[s:e],
        )


class TriBEDataModule(L.LightningDataModule):
    """
    DataModule para TriBEStyleModel.

    Args:
        vit_path:       Path a real_vit_features.pt.
        conformer_path: Path a real_conformer_features.pt.
        text_path:      Path a real_text_features.pt.
        bold_path:      Path a sub-XX.pt (fMRI filtrado).
        window_size:    TRs por ventana (default 67 ≈ 100s a TR=1.49s).
        stride:         Avance entre ventanas (default 5).
        hrf_delay:      Retraso HRF en segundos (default 5.0s).
        fmri_tr:        TR en segundos (default 1.49s).
        val_split:      Fracción del dataset para validación (default 0.1).
        batch_size:     Tamaño del batch (default 16).
        normalize_bold: Si True, z-score por parcela sobre el conjunto de train.
        normalize_feats: Si True, z-score por dimensión para cada modalidad.
    """

    def __init__(
        self,
        vit_path:       str,
        conformer_path: str,
        text_path:      str,
        bold_path:      str,
        window_size:    int   = 67,
        stride:         int   = 5,
        hrf_delay:      float = 5.0,
        fmri_tr:        float = 1.49,
        val_split:      float = 0.1,
        batch_size:     int   = 16,
        normalize_bold: bool  = True,
        normalize_feats: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str = None):
        # 1. Cargar los 3 tensores de features + BOLD
        vit       = torch.load(self.hparams.vit_path,       weights_only=True).float()
        conformer = torch.load(self.hparams.conformer_path, weights_only=True).float()
        text      = torch.load(self.hparams.text_path,      weights_only=True).float()
        bold      = torch.load(self.hparams.bold_path,      weights_only=True).float()

        print(
            f"TriBEDataModule — Tensores cargados:\n"
            f"  ViT:       {tuple(vit.shape)}\n"
            f"  Conformer: {tuple(conformer.shape)}\n"
            f"  Texto:     {tuple(text.shape)}\n"
            f"  BOLD:      {tuple(bold.shape)}"
        )

        # 2. Alineación HRF (igual que en el datamodule v2)
        aligner = HRFAligner(
            hrf_delay_seconds=self.hparams.hrf_delay,
            fmri_tr_seconds=self.hparams.fmri_tr,
        )
        # El aligner desplaza los features n_shift TRs hacia adelante
        # (features[t] predice bold[t + n_shift])
        vit,       bold_v  = aligner.align_stimulus_to_fmri(vit,       bold)
        conformer, bold_c  = aligner.align_stimulus_to_fmri(conformer, bold)
        text,      bold_t  = aligner.align_stimulus_to_fmri(text,      bold)

        # Todos deben retornar el mismo bold alineado
        assert bold_v.shape == bold_c.shape == bold_t.shape, (
            "Error: los BLDs alineados tienen shapes distintas."
        )
        bold = bold_v

        # 3. Normalización independiente por modalidad (z-score por dimensión)
        #    Evita que la modalidad con mayor varianza domine el gradient.
        if self.hparams.normalize_feats:
            for name, tensor in [("vit", vit), ("conformer", conformer), ("text", text)]:
                mean = tensor.mean(dim=0, keepdim=True)
                std  = tensor.std(dim=0, keepdim=True).clamp(min=1e-8)
                if name == "vit":
                    vit       = (vit - mean) / std
                elif name == "conformer":
                    conformer = (conformer - mean) / std
                else:
                    text      = (text - mean) / std

        # 4. Normalización BOLD
        if self.hparams.normalize_bold:
            bold_mean = bold.mean(dim=0, keepdim=True)
            bold_std  = bold.std(dim=0, keepdim=True).clamp(min=1e-8)
            bold = (bold - bold_mean) / bold_std

        # 5. Dataset con ventanas deslizantes on-the-fly
        full_dataset = _TriBESlidingWindowDataset(
            vit, conformer, text, bold,
            window_size=self.hparams.window_size,
            stride=self.hparams.stride,
        )
        total    = len(full_dataset)
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
            self.val_dataset   = full_dataset

        # Memoria estimada (solo tensores base, no ventanas materializadas)
        total_mb = sum(
            t.numel() * t.element_size() / (1024 ** 2)
            for t in [vit, conformer, text, bold]
        )
        print(
            f"  Ventanas: total={total}, train={train_size}, val={val_size}\n"
            f"  Memoria base: {total_mb:.1f} MB"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=0,
        )
