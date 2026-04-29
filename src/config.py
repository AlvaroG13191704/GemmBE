"""
Configuración centralizada de hiperparámetros para MicroTRIBE-Gemma.

Todos los valores de Gemma 4 están extraídos directamente del config.json oficial:
https://huggingface.co/google/gemma-4-E2B-it/raw/main/config.json

Los valores de TriBE v2 están tomados del paper original de Meta FAIR.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch


def get_device() -> torch.device:
    """
    Detección automática del mejor dispositivo disponible.
    
    Prioridad: CUDA (Colab GPU) → MPS (Apple Silicon) → CPU
    
    En Google Colab con TPU, PyTorch usa xla; si torch_xla está disponible,
    se puede usar `xm.xla_device()`. Para simplicidad, esta función cubre
    los casos más comunes (GPU Colab y MPS local).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    """
    Selecciona el dtype óptimo según el dispositivo.
    
    - CUDA: bfloat16 (soporte nativo en Ampere+, Colab T4 usa float16)
    - MPS: float32 (bfloat16 tiene soporte limitado en MPS)
    - CPU: float32
    """
    if device.type == "cuda":
        # T4 en Colab Free no soporta bf16 nativo, pero funciona vía emulación.
        # A100/H100 tienen soporte nativo.
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    elif device.type == "mps":
        # MPS en Apple Silicon: bfloat16 puede causar errores en algunas ops.
        # float16 es más seguro.
        return torch.float16
    return torch.float32


@dataclass
class ModelConfig:
    """
    Configuración completa del pipeline MicroTRIBE-Gemma.
    
    Attributes:
        model_id: Identificador de HuggingFace del modelo Gemma 4.
        gemma_hidden_size: Dimensión del hidden state del decoder de Gemma 4 E2B.
                          Extraído de text_config.hidden_size en config.json = 1536.
        bottleneck_size: Dimensión del cuello de botella antes del Subject Block.
                        Reduce de 1536 → 512 (ratio 3:1).
        num_vertices: Número de vértices corticales en la superficie fsaverage5.
                     TriBE v2 usa 20,484 vértices.
        fmri_sampling_rate_hz: Frecuencia de muestreo de la fMRI en Hz.
                              El TR típico del dataset Cam-CAN es ~1 segundo.
        hrf_delay_seconds: Retraso hemodinámico (HRF) en segundos.
                          La sangre tarda ~5s en reflejar la actividad neuronal.
        extraction_layer: Índice de la capa de hidden states a extraer.
                         -1 = última capa (default). Gemma 4 E2B tiene 35 capas.
        max_audio_seconds: Límite máximo de audio por ventana (limitación de Gemma 4).
        freeze_backbone: Si True, congela todos los parámetros de Gemma 4.
        
    Valores de referencia del config.json de Gemma 4 E2B:
        - text_config.hidden_size: 1536
        - text_config.num_hidden_layers: 35
        - vision_config.hidden_size: 768
        - audio_config.hidden_size: 1024
        - audio_config.output_proj_dims: 1536 (proyectado al espacio del texto)
    """
    # --- Gemma 4 ---
    model_id: str = "google/gemma-4-E2B-it"
    gemma_hidden_size: int = 1536
    
    # --- Architecture ---
    bottleneck_size: int = 512
    num_vertices: int = 0  # 0 = auto-detect from dataset. Set manually to override.
    extraction_layer: int = -1
    freeze_backbone: bool = True
    
    # --- Temporal ---
    fmri_sampling_rate_hz: float = 1.0
    hrf_delay_seconds: float = 5.0
    max_audio_seconds: float = 30.0
    
    # --- Device (auto-detected) ---
    _device: Optional[torch.device] = field(default=None, repr=False)
    _dtype: Optional[torch.dtype] = field(default=None, repr=False)
    
    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = get_device()
        return self._device
    
    @property
    def dtype(self) -> torch.dtype:
        if self._dtype is None:
            self._dtype = get_dtype(self.device)
        return self._dtype
    
    def override_device(self, device: str) -> None:
        """Permite forzar un dispositivo específico (útil para tests)."""
        self._device = torch.device(device)
        self._dtype = get_dtype(self._device)
