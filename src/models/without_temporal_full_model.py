"""
Full Model: Bottleneck(1536→512) + SubjectBlock(512→num_vertices).
"""

import torch
import torch.nn as nn
# pyrefly: ignore [missing-import]
from src.models.base_module import BrainEncodingModule
# pyrefly: ignore [missing-import]
from src.config import ModelConfig

class WithoutTemporalFullModel(BrainEncodingModule):
    """Modelo full SIN Transformer temporal (pointwise). Bottleneck 1536→512→num_vertices."""

    def __init__(self, **kwargs):
        super().__init__(model_name="without_temporal_full", **kwargs)
        self.config = ModelConfig()
        self.model = nn.Sequential(
            nn.Linear(self.config.gemma_hidden_size, self.config.bottleneck_size),
            nn.LayerNorm(self.config.bottleneck_size),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.config.bottleneck_size, self.hparams.num_vertices),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)
