"""
Configuración centralizada de hiperparámetros para GemmaBE.

Todos los valores de Gemma 4 están extraídos directamente del config.json oficial:
https://huggingface.co/google/gemma-4-E2B-it/raw/main/config.json

v3: Se añaden vit_hidden_size (SigLIP ViT-So400m) y conformer_hidden_size
(USM Conformer) para la extracción de embeddings pre-proyección.
"""

import torch
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    Configuración completa del pipeline GemmaBE.

    Attributes:
        model_id: Identificador de HuggingFace del modelo Gemma 4.
        gemma_hidden_size: Dimensión del hidden state del decoder = 1536.
        vit_hidden_size: Dimensión de salida del ViT SigLIP-So400m (pre-proyección).
                        Confirmado en config.json: vision_config.hidden_size = 1152.
        conformer_hidden_size: Dimensión de salida del Conformer USM (pre-proyección).
                              Confirmado en config.json: audio_config.hidden_size = 512.
        shared_dim: Dimensión compartida para las FFNs de cada modalidad en v3.
                   Cada rama (vit, conformer, text) se proyecta a shared_dim=512.
        bottleneck_size: Mantenido para retrocompatibilidad con modelos v2.
        num_vertices: Número de parcelas/vértices del atlas fMRI.
                     Algonauts 2025 usa 1,000 parcelas (Schaefer-1000).
        hrf_delay_seconds: Retraso hemodinámico (HRF) en segundos. 5.0 por defecto.
        max_audio_seconds: Límite máximo de audio por ventana.
        freeze_backbone: Si True, congela todos los parámetros de Gemma 4.
    """
    model_id: str = "google/gemma-4-E2B"
    # ── Dimensiones del decoder LLM ──────────────────────────────────────────
    gemma_hidden_size: int = 1536
    # ── Dimensiones pre-proyección (v3) ─────────────────────────────────────
    vit_hidden_size: int = 768         # Gemma4 ViT encoder output dim
    conformer_hidden_size: int = 1024  # Gemma4 Audio (Conformer) output dim
    shared_dim: int = 512              # Dim compartida tras cada FFN modal
    # ── Retrocompatibilidad v2 ───────────────────────────────────────────────
    bottleneck_size: int = 512
    # ── Parámetros fMRI / HRF ───────────────────────────────────────────────
    num_vertices: int = 1000
    hrf_delay_seconds: float = 5.0
    max_audio_seconds: float = 30.0
    freeze_backbone: bool = True

    @property
    def device(self) -> torch.device:
        """Dispositivo de cómputo (CUDA si está disponible, si no CPU)."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def dtype(self) -> torch.dtype:
        """Dtype: bfloat16 en CUDA, float32 en CPU."""
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32

    @property
    def transformer_dim(self) -> int:
        """Dimensión de entrada al Transformer en v3 (3 × shared_dim)."""
        return self.shared_dim * 3
