"""
WithoutTemporalFullModel — Ablation del Transformer temporal.

Mantiene el mismo bottleneck que TemporalFullModel pero reemplaza el
Transformer por un MLP pointwise de 2 capas ocultas. Esto hace la
ablation justa: la unica diferencia real es la arquitectura temporal.
"""

import torch
import torch.nn as nn
# pyrefly: ignore [missing-import]
from src.models.base_module import BrainEncodingModule
# pyrefly: ignore [missing-import]
from src.config import ModelConfig


class WithoutTemporalFullModel(BrainEncodingModule):
    """
    Modelo full SIN Transformer temporal (pointwise).

    Arquitectura:
        Bottleneck(1536 -> 512) -> LN -> GELU
        MLP pointwise: Linear(512 -> 512) -> LN -> GELU -> Dropout
                       Linear(512 -> 512) -> LN -> GELU -> Dropout
        Head: Linear(512 -> num_vertices)

    Parámetros: ~1.8M (vs 26.6M del temporal).
    """

    def __init__(self, **kwargs):
        super().__init__(model_name="without_temporal_full", **kwargs)
        self.config = ModelConfig()
        hidden = self.config.bottleneck_size  # 512

        # Bottleneck (identico al temporal)
        self.bottleneck = nn.Sequential(
            nn.Linear(self.config.gemma_hidden_size, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        # MLP pointwise de 2 capas ocultas (reemplaza el Transformer)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # Head (identico al temporal)
        self.head = nn.Linear(hidden, self.hparams.num_vertices)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.bottleneck(features)
        x = self.mlp(x)
        return self.head(x)
