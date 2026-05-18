"""
Temporal Full Model: Bottleneck + TemporalTransformer + SubjectBlock.

Procesa ventanas de TRs con self-attention temporal.
"""

import torch
import torch.nn as nn
# pyrefly: ignore [missing-import]
from src.models.base_module import BrainEncodingModule
# pyrefly: ignore [missing-import]
from src.config import ModelConfig
# pyrefly: ignore [missing-import]
from src.architecture.temporal_transformer import TemporalTransformerEncoder


class TemporalFullModel(BrainEncodingModule):
    """Modelo full con Transformer temporal entre bottleneck y salida."""

    def __init__(
        self,
        window_size: int = 67,
        transformer_layers: int = 8,
        transformer_heads: int = 8,
        transformer_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(model_name="full_transformer", **kwargs)
        self.window_size = window_size
        self.config = ModelConfig()

        self.bottleneck = nn.Sequential(
            nn.Linear(self.config.gemma_hidden_size, self.config.bottleneck_size),
            nn.LayerNorm(self.config.bottleneck_size),
            nn.GELU(),
        )
        self.transformer = TemporalTransformerEncoder(
            d_model=self.config.bottleneck_size,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            max_window=window_size * 2,
            dropout=transformer_dropout,
            num_subjects=0,
        )
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(self.config.bottleneck_size, self.hparams.num_vertices)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, W, 1536) ventana de TRs.
        Returns:
            predicted_bold: (B, W, num_vertices)
        """
        B, W, D = features.shape
        # Bottleneck per-timestep
        x = self.bottleneck(features.reshape(B * W, D)).reshape(B, W,self.config.bottleneck_size)
        # Transformer temporal
        x = self.transformer(x)
        x = self.dropout(x)
        # Head per-timestep
        out = self.head(x.reshape(B * W, self.config.bottleneck_size)).reshape(B, W, self.hparams.num_vertices)
        return out

    def training_step(self, batch, batch_idx):
        features, bold = batch  # (B, W, 1536), (B, W, num_vertices)
        pred = self(features)
        loss = nn.functional.mse_loss(pred, bold)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        features, bold = batch
        pred = self(features)
        loss = nn.functional.mse_loss(pred, bold)
        # Pearson por parcela promediando sobre timesteps y batch
        pred_flat = pred.reshape(-1, self.hparams.num_vertices)
        bold_flat = bold.reshape(-1, self.hparams.num_vertices)
        pearson = self._compute_pearson(pred_flat, bold_flat)
        avg_pearson = pearson.mean()

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/pearson", avg_pearson, on_epoch=True, prog_bar=True)
        self._last_val_pearson_map = pearson.detach().cpu()
        return {"val_loss": loss, "val_pearson": avg_pearson}

    def test_step(self, batch, batch_idx):
        features, bold = batch
        pred = self(features)
        loss = nn.functional.mse_loss(pred, bold)
        pred_flat = pred.reshape(-1, self.hparams.num_vertices)
        bold_flat = bold.reshape(-1, self.hparams.num_vertices)
        pearson = self._compute_pearson(pred_flat, bold_flat)
        avg_pearson = pearson.mean()

        self.log("test/loss", loss, on_epoch=True)
        self.log("test/pearson", avg_pearson, on_epoch=True)
        self._last_test_pearson_map = pearson.detach().cpu()
        return {"test_loss": loss, "test_pearson": avg_pearson}
