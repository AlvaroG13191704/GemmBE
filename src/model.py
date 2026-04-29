"""
MicroTribeGemma: Modelo principal que conecta Gemma 4 con la predicción cortical.

Arquitectura completa (Early Fusion):
    
    Estímulo (video + audio + texto)
       │
       ▼
    ┌─────────────────────────────────┐
    │  Gemma 4 E2B-it (CONGELADO)     │
    │  AutoModelForMultimodalLM       │
    │  output_hidden_states=True      │
    │                                 │
    │  Entrada:                       │
    │    - input_ids (texto)          │
    │    - pixel_values (video)       │
    │    - input_features (audio)     │
    │    - attention_mask             │
    │                                 │
    │  Salida: hidden_states[-1]      │
    │  Forma: (B, SeqLen, 1536)       │
    └─────────────┬───────────────────┘
                  │
                  ▼
    ┌─────────────────────────────────┐
    │  Temporal Pooling               │
    │  (B, SeqLen, 1536) → (B, 1536)  │
    └─────────────┬───────────────────┘
                  │
                  ▼
    ┌─────────────────────────────────┐
    │  Bottleneck                     │
    │  Linear(1536→512) + LN + GELU   │
    │  (B, 1536) → (B, 512)           │
    └─────────────┬───────────────────┘
                  │
                  ▼
    ┌─────────────────────────────────┐
    │  Subject Block                  │
    │  Linear(512→20484)              │
    │  (B, 512) → (B, 20484)          │
    └─────────────┬───────────────────┘
                  │
                  ▼
    Predicción BOLD (B, 20484)
"""

import torch
import torch.nn as nn
from typing import Optional

from src.config import ModelConfig
from src.temporal_alignment import TemporalPooling
from src.subject_block import Bottleneck, SubjectBlock, MultiSubjectBlock


class DummyGemmaBackbone(nn.Module):
    """
    Backbone simulado que replica las dimensiones de salida de Gemma 4 E2B.
    
    Se usa para validación de la arquitectura sin descargar el modelo real (~5GB).
    Genera hidden_states con la misma forma que Gemma 4: (B, SeqLen, 1536).
    
    Esto permite:
    1. Verificar que todas las capas downstream tienen shapes correctos.
    2. Correr tests sin GPU ni descarga del modelo.
    3. Desarrollar el pipeline de entrenamiento de forma rápida.
    """
    
    def __init__(self, hidden_size: int = 1536, num_layers: int = 35):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Una capa simple para generar embeddings de las dimensiones correctas
        self.embed = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, dummy_seq_len: int = 128, batch_size: int = 1, device: torch.device = None, dtype: torch.dtype = None):
        """Genera hidden states simulados."""
        hidden = torch.randn(batch_size, dummy_seq_len, self.hidden_size, device=device, dtype=dtype)
        # Simular la tupla de hidden_states (una por capa + embedding)
        hidden_states = tuple(
            torch.randn_like(hidden) for _ in range(self.num_layers + 1)
        )
        return hidden_states


class MicroTribeGemma(nn.Module):
    """
    Pipeline completo: Gemma 4 → Pooling → Bottleneck → SubjectBlock → BOLD.
    
    Modos de operación:
    
    1. dummy=True (default para desarrollo):
       - No descarga el modelo de Gemma 4.
       - Usa DummyGemmaBackbone para validar shapes.
       - Requiere 0 VRAM extra.
    
    2. dummy=False (producción):
       - Descarga y carga Gemma 4 E2B-it desde HuggingFace.
       - Congela todos los parámetros de Gemma.
       - Requiere ~5GB VRAM (bfloat16) o ~3GB (int8).
    
    Args:
        config: Instancia de ModelConfig con todos los hiperparámetros.
        subject_ids: Lista de IDs de sujeto para MultiSubjectBlock.
                    Si es None, usa un SubjectBlock genérico.
        pooling_mode: 'mean' o 'conv1d' para TemporalPooling.
        dummy: Si True, usa backbone simulado en lugar de Gemma 4 real.
    """
    
    def __init__(
        self,
        config: ModelConfig = None,
        subject_ids: list[str] = None,
        pooling_mode: str = "conv1d",
        dummy: bool = True,
    ):
        super().__init__()
        self.config = config or ModelConfig()
        self.dummy = dummy
        
        device = self.config.device
        dtype = self.config.dtype
        
        print("╔══════════════════════════════════════════════╗")
        print("║   MicroTRIBE-Gemma v0.1                     ║")
        print(f"║   Device: {str(device):<35s}║")
        print(f"║   Dtype:  {str(dtype):<35s}║")
        print(f"║   Mode:   {'DUMMY (sin modelo real)' if dummy else 'REAL (Gemma 4 E2B-it)':<35s}║")
        print("╚══════════════════════════════════════════════╝")
        
        # --- BACKBONE ---
        if dummy:
            self.backbone = DummyGemmaBackbone(
                hidden_size=self.config.gemma_hidden_size,
            )
            self.processor = None
        else:
            self._load_gemma(device, dtype)
        
        # --- TEMPORAL POOLING ---
        self.temporal_pooling = TemporalPooling(
            hidden_size=self.config.gemma_hidden_size,
            mode=pooling_mode,
            dtype=dtype,
        )
        
        # --- BOTTLENECK ---
        self.bottleneck = Bottleneck(
            input_size=self.config.gemma_hidden_size,
            output_size=self.config.bottleneck_size,
            dtype=dtype,
        )
        
        # --- SUBJECT BLOCK ---
        if subject_ids:
            self.subject_block = MultiSubjectBlock(
                subject_ids=subject_ids,
                input_size=self.config.bottleneck_size,
                num_vertices=self.config.num_vertices,
                dtype=dtype,
            )
            self._multi_subject = True
        else:
            self.subject_block = SubjectBlock(
                input_size=self.config.bottleneck_size,
                num_vertices=self.config.num_vertices,
                dtype=dtype,
            )
            self._multi_subject = False
    
    def _load_gemma(self, device: torch.device, dtype: torch.dtype) -> None:
        """Carga Gemma 4 E2B-it desde HuggingFace y congela sus parámetros."""
        from transformers import AutoProcessor, AutoModelForMultimodalLM
        
        print(f"  Descargando {self.config.model_id}...")
        
        self.processor = AutoProcessor.from_pretrained(self.config.model_id)
        
        # device_map="auto" distribuye el modelo entre GPU/CPU automáticamente
        self.backbone = AutoModelForMultimodalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
            device_map="auto" if device.type == "cuda" else None,
        )
        
        # Si no es CUDA con device_map, mover manualmente al dispositivo
        if device.type != "cuda":
            self.backbone = self.backbone.to(device)
        
        # Congelar TODOS los parámetros de Gemma 4
        if self.config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"  ✅ Gemma 4 congelado ({sum(p.numel() for p in self.backbone.parameters()):,} parámetros)")
        
        # Poner en modo evaluación (desactiva dropout, etc.)
        self.backbone.eval()
    
    def extract_hidden_states(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        pixel_values: torch.Tensor = None,
        input_features: torch.Tensor = None,
        dummy_seq_len: int = 128,
        dummy_batch_size: int = 1,
    ) -> torch.Tensor:
        """
        Extrae los hidden states de Gemma 4 (o del backbone dummy).
        
        En modo real:
            Pasa los tensores por Gemma 4 con output_hidden_states=True
            y extrae la capa especificada por config.extraction_layer.
        
        En modo dummy:
            Genera tensores aleatorios con las dimensiones correctas.
        
        Returns:
            hidden_states: (Batch, SeqLen, 1536)
        """
        if self.dummy:
            device = self.config.device
            dtype = self.config.dtype
            hs = self.backbone(
                dummy_seq_len=dummy_seq_len,
                batch_size=dummy_batch_size,
                device=device,
                dtype=dtype,
            )
            return hs[self.config.extraction_layer]
        
        # --- Modo REAL con Gemma 4 ---
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                input_features=input_features,
                output_hidden_states=True,
                return_dict=True,
            )
        
        # hidden_states es una tupla: (embedding, capa_1, ..., capa_35)
        # extraction_layer=-1 toma la última capa del decoder
        return outputs.hidden_states[self.config.extraction_layer]
    
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        pixel_values: torch.Tensor = None,
        input_features: torch.Tensor = None,
        subject_id: str = None,
        dummy_seq_len: int = 128,
        dummy_batch_size: int = 1,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass completo del pipeline.
        
        Args:
            input_ids: (B, TextLen) — tokens de texto.
            attention_mask: (B, TotalSeqLen) — máscara combinada.
            pixel_values: (B, NumFrames, C, H, W) — frames de video.
            input_features: (B, AudioLen, AudioDim) — espectrograma MEL.
            subject_id: ID del sujeto (requerido si multi-subject).
            dummy_seq_len: Longitud de secuencia para modo dummy.
            dummy_batch_size: Batch size para modo dummy.
        
        Returns:
            dict con:
                'predicted_bold': (B, 20484) — actividad BOLD predicha.
                'hidden_states': (B, SeqLen, 1536) — embeddings crudos.
                'pooled': (B, 1536) — embedding pooled.
                'compressed': (B, 512) — embedding comprimido.
        """
        # A. Extraer hidden states de Gemma 4
        hidden_states = self.extract_hidden_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            input_features=input_features,
            dummy_seq_len=dummy_seq_len,
            dummy_batch_size=dummy_batch_size,
        )
        
        # B. Temporal Pooling: (B, SeqLen, 1536) → (B, 1536)
        pooled = self.temporal_pooling(hidden_states, attention_mask)
        
        # C. Bottleneck: (B, 1536) → (B, 512)
        compressed = self.bottleneck(pooled)
        
        # D. Subject Block: (B, 512) → (B, 20484)
        if self._multi_subject:
            if subject_id is None:
                raise ValueError(
                    "subject_id es requerido cuando se usa MultiSubjectBlock. "
                    "Pasa subject_id='sub-XX' al forward()."
                )
            predicted_bold = self.subject_block(compressed, subject_id)
        else:
            predicted_bold = self.subject_block(compressed)
        
        return {
            "predicted_bold": predicted_bold,
            "hidden_states": hidden_states,
            "pooled": pooled,
            "compressed": compressed,
        }
    
    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """
        Retorna solo los parámetros entrenables (no los de Gemma 4).
        
        Útil para configurar el optimizador:
            optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-4)
        """
        trainable = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable.append(param)
        return trainable
    
    def print_parameter_summary(self) -> None:
        """Imprime un resumen de parámetros totales vs entrenables."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        
        print("\n📊 Resumen de Parámetros:")
        print(f"  Total:       {total:>15,}")
        print(f"  Entrenables: {trainable:>15,} ({trainable/max(total,1)*100:.1f}%)")
        print(f"  Congelados:  {frozen:>15,} ({frozen/max(total,1)*100:.1f}%)")
        
        print("\n  Desglose entrenables:")
        for name, module in self.named_children():
            if name == "backbone":
                continue
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if params > 0:
                print(f"    {name}: {params:,}")
