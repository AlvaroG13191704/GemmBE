"""
TailModel: Modelo ligero para entrenamiento offline.

Este modelo es la "cola" del pipeline — todo lo que viene DESPUÉS de Gemma 4.
Se usa cuando los features ya fueron pre-extraídos y guardados en disco.

Comparación:
    MicroTribeGemma (modelo completo):
        Gemma 4 (2.3B params) → Pooling → Bottleneck → SubjectBlock
        Requiere ~5GB VRAM, lento
    
    TailModel (este archivo):
        Bottleneck → Dropout → SubjectBlock
        Requiere ~500MB VRAM, rapidísimo 🚀

El Temporal Pooling NO se incluye aquí porque ya se aplicó durante
la extracción offline. Los features de entrada ya están pooled: (B, 1536).

Arquitectura:
    Features pre-extraídos (B, 1536)
        │
        ▼
    ┌─────────────────────────────────┐
    │  Bottleneck                     │
    │  Linear(1536→512) + LN + GELU  │
    │  (B, 1536) → (B, 512)          │
    └─────────────┬───────────────────┘
                  │
                  ▼
    ┌─────────────────────────────────┐
    │  Dropout(p=0.3)                 │
    │  Regularización anti-overfit    │
    │  (solo activo en training)      │
    └─────────────┬───────────────────┘
                  │
                  ▼
    ┌─────────────────────────────────┐
    │  Subject Block                  │
    │  Linear(512→num_vertices)       │
    │  (B, 512) → (B, num_vertices)  │
    └─────────────┬───────────────────┘
                  │
                  ▼
    Predicción BOLD (B, num_vertices)

Parámetros totales: ~122M con MNI152 (238,955 vóxeles) / ~11M con fsaverage5 (20,484)
"""

import torch
import torch.nn as nn

from src.config import ModelConfig
from src.subject_block import Bottleneck, SubjectBlock, MultiSubjectBlock


class TailModel(nn.Module):
    """
    Modelo ligero para entrenamiento con features pre-extraídos.
    
    Solo incluye Bottleneck + SubjectBlock. No carga Gemma 4 ni
    TemporalPooling (ambos ya se usaron en la extracción offline).
    
    Args:
        config: ModelConfig con hiperparámetros.
        subject_ids: Lista de IDs de sujeto. Si es None, usa un solo SubjectBlock.
    """
    
    def __init__(
        self,
        config: ModelConfig = None,
        subject_ids: list[str] = None,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        self.config = config or ModelConfig()
        
        device = self.config.device
        dtype = self.config.dtype
        
        print("╔══════════════════════════════════════════════╗")
        print("║   TailModel (Entrenamiento Offline)         ║")
        print(f"║   Device: {str(device):<35s}║")
        print(f"║   Dtype:  {str(dtype):<35s}║")
        print(f"║   Dropout: p={dropout_p:<31s}║")
        print("╚══════════════════════════════════════════════╝")
        
        # --- BOTTLENECK ---
        self.bottleneck = Bottleneck(
            input_size=self.config.gemma_hidden_size,
            output_size=self.config.bottleneck_size,
            dtype=dtype,
        )
        
        # --- DROPOUT (Regularización contra sobreajuste) ---
        # Con ~122M params y ~943 muestras, el ratio datos/params es muy bajo.
        # Este Dropout actúa como el equivalente funcional del "Modality Dropout"
        # de TriBE v2: obliga al SubjectBlock a no depender de un subconjunto
        # fijo de features del bottleneck.
        # Solo se activa durante training (model.train()); en eval es no-op.
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
        else:
            self.subject_block = SubjectBlock(
                input_size=self.config.bottleneck_size,
                num_vertices=self.config.num_vertices,
                dtype=dtype,
            )
            self._multi_subject = False
            self._subject_ids = []
    
    def forward(
        self,
        features: torch.Tensor,
        subject_id: str = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass del modelo ligero.
        
        Args:
            features: (B, 1536) — features pre-extraídos de Gemma 4.
            subject_id: ID del sujeto (requerido si multi-subject).
        
        Returns:
            dict con:
                'predicted_bold': (B, num_vertices) — predicción BOLD.
                'compressed': (B, 512) — embedding comprimido.
        """
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
    
    def print_parameter_summary(self) -> None:
        """Imprime resumen de parámetros."""
        total = sum(p.numel() for p in self.parameters())
        
        print("\n📊 Resumen de Parámetros (TailModel):")
        print(f"  Total entrenables: {total:>12,}")
        
        bn_params = sum(p.numel() for p in self.bottleneck.parameters())
        sb_params = sum(p.numel() for p in self.subject_block.parameters())
        print(f"    Bottleneck:    {bn_params:>12,}")
        print(f"    SubjectBlock:  {sb_params:>12,}")
        
        if self._multi_subject:
            per_subject = sb_params // len(self._subject_ids)
            print(f"    Per subject:   {per_subject:>12,}")
        
        # Estimación de VRAM
        bytes_per_param = 2 if self.config.dtype in (torch.float16, torch.bfloat16) else 4
        vram_mb = (total * bytes_per_param) / (1024 * 1024)
        print(f"\n  💾 VRAM estimada: {vram_mb:.1f} MB")
        print("     (vs ~5,000 MB con Gemma 4 completo)")
