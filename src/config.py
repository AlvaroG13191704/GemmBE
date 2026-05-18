"""
Configuración centralizada de hiperparámetros para GemmaBE.

Todos los valores de Gemma 4 están extraídos directamente del config.json oficial:
https://huggingface.co/google/gemma-4-E2B-it/raw/main/config.json

Nota: device y dtype ya NO están aquí. PyTorch Lightning maneja el hardware
automáticamente via Trainer(accelerator=..., precision=...).
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    Configuración completa del pipeline GemmaBE.
    
    Attributes:
        model_id: Identificador de HuggingFace del modelo Gemma 4.
        gemma_hidden_size: Dimensión del hidden state del decoder = 1536.
        bottleneck_size: Dimensión del cuello de botella. Reduce de 1536 → 512.
        num_vertices: Número de parcelas/vértices del atlas fMRI.
                     Algonauts 2025 usa 1,000 parcelas (Schaefer-1000).
        hrf_delay_seconds: Retraso hemodinámico (HRF) en segundos. 5.0 por defecto.
                          0.0 para ablation "no_hrf".
        max_audio_seconds: Límite máximo de audio por ventana.
        freeze_backbone: Si True, congela todos los parámetros de Gemma 4.
    """
    model_id: str = "google/gemma-4-E2B-it"
    gemma_hidden_size: int = 1536
    bottleneck_size: int = 512
    num_vertices: int = 1000
    hrf_delay_seconds: float = 5.0
    max_audio_seconds: float = 30.0
    freeze_backbone: bool = True
