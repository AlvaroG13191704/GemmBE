"""
Base LightningModule para Brain Encoding.

Contiene la lógica compartida: training/validation/test steps,
optimizadores, logging de métricas, y guardado de hiperparámetros.

Toda la métrica principal es Pearson correlation per parcel,
que es el estándar en brain encoding (TriBE v2, Algonauts).
"""

import torch
from torch import nn
import lightning as L


class BrainEncodingModule(L.LightningModule):
    """
    LightningModule base para todos los modelos de brain encoding.

    Args:
        model_name: Nombre del modelo (temporal_full, without_temporal_full, no_hrf, ridge).
        stimulus_type: "multimodal" o "textonly".
        subject_id: ID del sujeto (ej: "sub-01").
        num_vertices: Número de parcelas (1000).
        lr: Learning rate.
        weight_decay: Weight decay para AdamW.
        max_epochs: Épocas totales (para scheduler cosine).
    """

    def __init__(
        self,
        model_name: str,
        stimulus_type: str,
        subject_id: str,
        num_vertices: int = 1000,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        max_epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.stimulus_type = stimulus_type
        self.subject_id = subject_id
        self.num_vertices = num_vertices

        # El modelo específico se define en las subclases
        self.model = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward: features (B, 1536) → predicted_bold (B, num_vertices)."""
        return self.model(features)

    def _compute_pearson(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Pearson correlation per parcel. Retorna tensor (num_vertices,)."""
        p = pred - pred.mean(dim=0, keepdim=True)
        t = target - target.mean(dim=0, keepdim=True)
        num = (p * t).sum(dim=0)
        den = p.norm(dim=0) * t.norm(dim=0) + 1e-8
        return num / den

    def training_step(self, batch, batch_idx):
        features, bold = batch
        pred = self(features)
        loss = nn.functional.mse_loss(pred, bold)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        features, bold = batch
        pred = self(features)
        loss = nn.functional.mse_loss(pred, bold)
        pearson = self._compute_pearson(pred, bold)
        avg_pearson = pearson.mean()

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/pearson", avg_pearson, on_epoch=True, prog_bar=True)
        self.log("val/pearson>0.15", (pearson > 0.15).float().mean(), on_epoch=True)

        # Guardar pearson map para el callback
        self._last_val_pearson_map = pearson.detach().cpu()
        return {"val_loss": loss, "val_pearson": avg_pearson}

    def test_step(self, batch, batch_idx):
        features, bold = batch
        pred = self(features)
        loss = nn.functional.mse_loss(pred, bold)
        pearson = self._compute_pearson(pred, bold)
        avg_pearson = pearson.mean()

        self.log("test/loss", loss, on_epoch=True)
        self.log("test/pearson", avg_pearson, on_epoch=True)

        self._last_test_pearson_map = pearson.detach().cpu()
        return {"test_loss": loss, "test_pearson": avg_pearson}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
