"""
TailModel: Modelo ligero para entrenamiento offline.

Este modelo es la "cola" del pipeline — todo lo que viene DESPUÉS de Gemma 4.
Se usa cuando los features ya fueron pre-extraídos y guardados en disco.

Soporta DOS modos de operación:

1. Pointwise (default):
     Bottleneck → Dropout → SubjectBlock
     Entrada: (B, 1536) → Salida: (B, num_vertices)
     Cada TR se procesa independientemente.

2. Temporal (TriBE-style):
     Bottleneck → TemporalTransformer → Dropout → SubjectBlock
     Entrada: (B, W, 1536) → Salida: (B, W, num_vertices)
     Los TRs de una ventana se comunican entre sí via self-attention.
     El Transformer aprende la dinámica hemodinámica de forma adaptativa.

Arquitectura Temporal:
    Features ventana (B, W, 1536)
        │
        ▼
    ┌─────────────────────────────────┐
    │  Bottleneck (per-timestep)      │
    │  Linear(1536→512) + LN + GELU  │
    │  (B, W, 1536) → (B, W, 512)   │
    └─────────────┬───────────────────┘
                  │
                  ▼
    ┌─────────────────────────────────┐
    │  Temporal Transformer           │
    │  8 layers × 8 heads             │
    │  + Positional Embedding         │
    │  + Subject Embedding            │
    │  (B, W, 512) → (B, W, 512)    │
    └─────────────┬───────────────────┘
                  │
                  ▼
    ┌─────────────────────────────────┐
    │  Dropout(p=0.1)                 │
    └─────────────┬───────────────────┘
                  │
                  ▼
    ┌─────────────────────────────────┐
    │  Subject Block (per-timestep)   │
    │  Linear(512→num_vertices)       │
    │  (B, W, 512) → (B, W, n_verts)│
    └─────────────┬───────────────────┘
                  │
                  ▼
    Predicción BOLD (B, W, num_vertices)
"""

import torch
import torch.nn as nn

from src.config import ModelConfig
from src.subject_block import Bottleneck, SubjectBlock, MultiSubjectBlock
from src.temporal_transformer import TemporalTransformerEncoder


class TailModel(nn.Module):
    """
    Modelo ligero para entrenamiento con features pre-extraídos.

    Soporta modo 'pointwise' (TR independiente) y 'temporal' (ventana con
    Transformer). El modo se selecciona con el parámetro `temporal`.

    Args:
        config: ModelConfig con hiperparámetros.
        subject_ids: Lista de IDs de sujeto. Si es None, usa un solo SubjectBlock.
        dropout_p: Dropout probability.
        temporal: Si True, usa el Temporal Transformer entre Bottleneck y SubjectBlock.
    """

    def __init__(
        self,
        config: ModelConfig = None,
        subject_ids: list[str] = None,
        dropout_p: float = 0.3,
        temporal: bool = False,
    ):
        super().__init__()
        self.config = config or ModelConfig()
        self.temporal = temporal

        device = self.config.device
        dtype = self.config.dtype

        mode_str = "Temporal (Transformer)" if temporal else "Pointwise"
        print("╔══════════════════════════════════════════════╗")
        print(f"║   TailModel — {mode_str:<30s}║")
        print(f"║   Device: {str(device):<35s}║")
        print(f"║   Dtype:  {str(dtype):<35s}║")
        if temporal:
            print(f"║   Window: {self.config.window_size_trs} TRs"
                  f" ({self.config.window_size_trs * 1.49:.0f}s)"
                  f"{'':>18s}║")
            print(f"║   Layers: {self.config.transformer_layers}"
                  f"  Heads: {self.config.transformer_heads}"
                  f"{'':>20s}║")
        else:
            print(f"║   Dropout: p={str(dropout_p):<31s}║")
        print("╚══════════════════════════════════════════════╝")

        # --- BOTTLENECK ---
        self.bottleneck = Bottleneck(
            input_size=self.config.gemma_hidden_size,
            output_size=self.config.bottleneck_size,
            dtype=dtype,
        )

        # --- TEMPORAL TRANSFORMER (solo en modo temporal) ---
        if temporal:
            num_subjects = len(subject_ids) if subject_ids else 0
            self.transformer = TemporalTransformerEncoder(
                d_model=self.config.bottleneck_size,
                nhead=self.config.transformer_heads,
                num_layers=self.config.transformer_layers,
                max_window=self.config.window_size_trs * 2,  # margen
                dropout=self.config.transformer_dropout,
                num_subjects=num_subjects,
            )
            # En modo temporal, usar el dropout del Transformer (más bajo)
            self.dropout = nn.Dropout(p=self.config.transformer_dropout)
        else:
            self.transformer = None
            self.dropout = nn.Dropout(p=dropout_p)

        # --- SUBJECT BLOCK ---
        if subject_ids:
            self.subject_block = MultiSubjectBlock(
                subject_ids=subject_ids,
                input_size=self.config.bottleneck_size,
                num_vertices=self.config.num_vertices,
                dtype=dtype,
            )
            self._multi_subject = True
            self._subject_ids = subject_ids
            # Mapeo de subject_id → índice numérico para subject embedding
            self._subject_to_idx = {sid: i for i, sid in enumerate(subject_ids)}
        else:
            self.subject_block = SubjectBlock(
                input_size=self.config.bottleneck_size,
                num_vertices=self.config.num_vertices,
                dtype=dtype,
            )
            self._multi_subject = False
            self._subject_ids = []
            self._subject_to_idx = {}

    def forward(
        self,
        features: torch.Tensor,
        subject_id: str = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass del modelo.

        Modo pointwise:
            features: (B, 1536) → predicted_bold: (B, num_vertices)

        Modo temporal:
            features: (B, W, 1536) → predicted_bold: (B, W, num_vertices)

        Args:
            features: Features pre-extraídos de Gemma 4.
            subject_id: ID del sujeto (requerido si multi-subject).

        Returns:
            dict con:
                'predicted_bold': Predicción BOLD.
                'compressed': Embedding comprimido post-bottleneck.
        """
        if self.temporal:
            return self._forward_temporal(features, subject_id)
        else:
            return self._forward_pointwise(features, subject_id)

    def _forward_pointwise(
        self,
        features: torch.Tensor,
        subject_id: str = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass para modo pointwise: (B, 1536) → (B, num_vertices)."""
        # A. Bottleneck: (B, 1536) → (B, 512)
        compressed = self.bottleneck(features)

        # B. Dropout: regularización (solo activo en training)
        compressed = self.dropout(compressed)

        # C. Subject Block: (B, 512) → (B, num_vertices)
        if self._multi_subject:
            if subject_id is None:
                raise ValueError(
                    "subject_id es requerido en modo multi-subject. "
                    f"Sujetos disponibles: {self._subject_ids}"
                )
            predicted_bold = self.subject_block(compressed, subject_id)
        else:
            predicted_bold = self.subject_block(compressed)

        return {
            "predicted_bold": predicted_bold,
            "compressed": compressed,
        }

    def _forward_temporal(
        self,
        features: torch.Tensor,
        subject_id: str = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass para modo temporal: (B, W, 1536) → (B, W, num_vertices)."""
        B, W, D = features.shape

        # A. Bottleneck per-timestep: (B, W, 1536) → (B, W, 512)
        #    Reshape a (B*W, 1536), aplicar bottleneck, reshape a (B, W, 512)
        features_flat = features.reshape(B * W, D)
        compressed_flat = self.bottleneck(features_flat)
        compressed = compressed_flat.reshape(B, W, -1)  # (B, W, 512)

        # B. Temporal Transformer: (B, W, 512) → (B, W, 512)
        #    Los TRs intercambian información temporal via self-attention
        subject_idx = self._subject_to_idx.get(subject_id) if subject_id else None
        compressed = self.transformer(compressed, subject_idx=subject_idx)

        # C. Dropout
        compressed = self.dropout(compressed)

        # D. Subject Block per-timestep: (B, W, 512) → (B, W, num_vertices)
        compressed_flat = compressed.reshape(B * W, -1)
        if self._multi_subject:
            if subject_id is None:
                raise ValueError(
                    "subject_id es requerido en modo multi-subject. "
                    f"Sujetos disponibles: {self._subject_ids}"
                )
            predicted_flat = self.subject_block(compressed_flat, subject_id)
        else:
            predicted_flat = self.subject_block(compressed_flat)

        predicted_bold = predicted_flat.reshape(B, W, -1)  # (B, W, num_vertices)

        return {
            "predicted_bold": predicted_bold,
            "compressed": compressed,
        }

    def print_parameter_summary(self) -> None:
        """Imprime resumen de parámetros."""
        total = sum(p.numel() for p in self.parameters())

        print("\nResumen de Parámetros (TailModel):""")
        print(f"  Modo: {'Temporal' if self.temporal else 'Pointwise'}")
        print(f"  Total entrenables: {total:>12,}")

        bn_params = sum(p.numel() for p in self.bottleneck.parameters())
        sb_params = sum(p.numel() for p in self.subject_block.parameters())
        print(f"    Bottleneck:    {bn_params:>12,}")

        if self.temporal and self.transformer is not None:
            tf_params = sum(p.numel() for p in self.transformer.parameters())
            print(f"    Transformer:   {tf_params:>12,}")

        print(f"    SubjectBlock:  {sb_params:>12,}")

        if self._multi_subject:
            per_subject = sb_params // len(self._subject_ids)
            print(f"    Per subject:   {per_subject:>12,}")

        # Estimación de VRAM
        bytes_per_param = 2 if self.config.dtype in (torch.float16, torch.bfloat16) else 4
        vram_mb = (total * bytes_per_param) / (1024 * 1024)
        print(f"\n  VRAM estimada: {vram_mb:.1f} MB")
