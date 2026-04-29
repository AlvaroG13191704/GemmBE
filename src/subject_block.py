"""
Subject Block: Proyección del espacio latente al espacio cortical.

Este módulo resuelve el Reto 2 del paper TriBE v2 — la Explosión de Dimensionalidad.

En TriBE v2:
    - El espacio latente concatenado era de 1,152 dimensiones.
    - Un Subject Block (capa lineal) proyectaba a 20,484 vértices (fsaverage5).
    - Parámetros: 1,152 x 20,484 = ~23.6M por sujeto.

En MicroTRIBE-Gemma (Dataset Sherlock, espacio MNI152):
    - El hidden state de Gemma 4 es de 1,536 dimensiones.
    - Target: 238,955 vóxeles volumétricos MNI152.
    - Sin bottleneck: 1,536 x 238,955 = ~367M por sujeto.
    - Con bottleneck (512): 512 x 238,955 = ~122M por sujeto.
    
La capa Bottleneck reduce de 1536 → 512 antes de proyectar a la corteza,
ahorrando ~245M de parámetros por sujeto.
"""

import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    """
    Capa de cuello de botella que comprime el embedding de Gemma 4.
    
    Pipeline: Linear(1536→512) → LayerNorm → GELU
    
    LayerNorm estabiliza el entrenamiento al normalizar las activaciones.
    GELU introduce no-linealidad (estándar en transformers modernos).
    
    Args:
        input_size: Dimensión de entrada (1536 = Gemma 4 hidden_size).
        output_size: Dimensión de salida (512 = bottleneck_size).
        dtype: Tipo de dato numérico.
    """
    
    def __init__(
        self,
        input_size: int = 1536,
        output_size: int = 512,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, output_size, dtype=dtype),
            nn.LayerNorm(output_size, dtype=dtype),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, input_size) — embedding pooled de Gemma 4.
        
        Returns:
            (Batch, output_size) — embedding comprimido.
        """
        return self.net(x)


class SubjectBlock(nn.Module):
    """
    Proyección lineal del espacio latente al espacio cortical para UN sujeto.
    
    Mapea el vector comprimido de 512 dimensiones a los N vóxeles/vértices
    del espacio cortical del dataset.
    
    Cada vóxel/vértice corresponde a un punto en la corteza cerebral.
    La salida es la predicción de la señal BOLD (actividad hemodinámica)
    en cada punto.
    
    En TriBE v2 este era literalmente una sola capa linear:
        y = Wx + b
        donde W ∈ R^{num_vertices x 512}, b ∈ R^{num_vertices}
    
    Configuraciones:
        - fsaverage5 (TriBE v2):   num_vertices = 20,484   → ~10.5M params
        - MNI152 (Sherlock):       num_vertices = 238,955  → ~122M params
    
    Args:
        input_size: Dimensión de entrada (512 = bottleneck_size).
        num_vertices: Número de vóxeles/vértices corticales.
        dtype: Tipo de dato numérico.
    """
    
    def __init__(
        self,
        input_size: int = 512,
        num_vertices: int = 238_955,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.projection = nn.Linear(input_size, num_vertices, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, input_size) — embedding comprimido del bottleneck.
        
        Returns:
            (Batch, num_vertices) — predicción BOLD por vértice cortical.
        """
        return self.projection(x)


class MultiSubjectBlock(nn.Module):
    """
    Colección de Subject Blocks para múltiples sujetos.
    
    En TriBE v2, cada sujeto tiene su propia capa linear de proyección,
    pero todos comparten el mismo backbone (Gemma 4 congelado) y bottleneck.
    
    Esto permite:
    1. Entrenar un modelo que generalice por sujeto (zero-shot).
    2. Fine-tune per-subject con pocas muestras (few-shot).
    3. Evaluar generalización cross-subject.
    
    Uso:
        multi_block = MultiSubjectBlock(["sub-01", "sub-02", "sub-03"])
        bold_pred = multi_block(compressed_embedding, subject_id="sub-01")
    
    Args:
        subject_ids: Lista de identificadores de sujeto.
        input_size: Dimensión de entrada del bottleneck.
        num_vertices: Vértices corticales.
        dtype: Tipo de dato numérico.
    """
    
    def __init__(
        self,
        subject_ids: list[str],
        input_size: int = 512,
        num_vertices: int = 238_955,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.subject_blocks = nn.ModuleDict({
            sid: SubjectBlock(input_size, num_vertices, dtype=dtype)
            for sid in subject_ids
        })
    
    def forward(self, x: torch.Tensor, subject_id: str) -> torch.Tensor:
        """
        Forward pass para un sujeto específico.
        
        Args:
            x: (Batch, input_size) — embedding comprimido.
            subject_id: Identificador del sujeto (e.g., "sub-01").
        
        Returns:
            (Batch, num_vertices) — predicción BOLD.
        
        Raises:
            KeyError: Si el subject_id no existe en el modelo.
        """
        if subject_id not in self.subject_blocks:
            available = list(self.subject_blocks.keys())
            raise KeyError(
                f"Sujeto '{subject_id}' no encontrado. "
                f"Sujetos disponibles: {available}"
            )
        return self.subject_blocks[subject_id](x)
    
    def get_trainable_params(self, subject_id: str = None) -> list[nn.Parameter]:
        """
        Retorna los parámetros entrenables de uno o todos los sujetos.
        
        Útil para configurar optimizadores per-subject.
        """
        if subject_id:
            return list(self.subject_blocks[subject_id].parameters())
        return list(self.parameters())
