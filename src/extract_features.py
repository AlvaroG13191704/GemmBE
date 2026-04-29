"""
Extractor Offline de Features de Gemma 4.

=== EL TRUCO DE ORO ===

Gemma 4 está CONGELADO. Sus pesos nunca cambian durante el entrenamiento.
Esto significa que para un mismo estímulo (video + audio), Gemma 4 SIEMPRE
produce el mismo hidden_state. Entonces, ¿para qué recalcularlo en cada época?

Flujo LENTO (sin extracción offline):
    Época 1: Cargar video → Gemma 4 → Pooling → Bottleneck → SubjectBlock → Loss
    Época 2: Cargar video → Gemma 4 → Pooling → Bottleneck → SubjectBlock → Loss
    ...
    Época 100: Cargar video → Gemma 4 → Pooling → Bottleneck → SubjectBlock → Loss
    
    → 100 forward passes de Gemma 4 (~5GB VRAM, ~2.3B params) x N muestras
    → LENTÍSIMO 🐌

Flujo RÁPIDO (con extracción offline):
    PRE-PROCESO (una sola vez):
        Cargar video → Gemma 4 → Pooling → Guardar tensor (1536,) en disco
    
    ENTRENAMIENTO (100 épocas):
        Época 1: Cargar tensor (1536,) → Bottleneck → SubjectBlock → Loss
        Época 2: Cargar tensor (1536,) → Bottleneck → SubjectBlock → Loss
        ...
        Época 100: Cargar tensor (1536,) → Bottleneck → SubjectBlock → Loss
    
    → 1 forward pass de Gemma 4 (pre-proceso)
    → 100 épocas con solo 11M params, sin Gemma en memoria
    → ~1000x MÁS RÁPIDO 🚀

Uso:
    # Paso 1: Extraer features offline (una sola vez)
    python -m src.extract_features --dataset_dir ./dataset --output_dir ./data/features
    
    # Paso 2: Entrenar solo la cola (muchas veces, experimentar)
    python train.py --features_dir ./data/features --subject sub-01

Mejoras v2 (Multi-Layer + LayerNorm + Sliding Window + Global-Only + 2Hz):
    - Extrae SOLO de capas de ATENCIÓN GLOBAL (no locales).
    - Gemma 4 E2B sigue un patrón 4:1 (4 sliding window + 1 global).
    - Las capas locales solo ven 512 tokens (~7s); las globales ven todo.
    - Capas extraídas: 19, 24, 29, 34 (globales en la mitad profunda).
    - LayerNorm PRE-POOLING: Normaliza cada token antes de promediar,
      evitando que los ~280 tokens de imagen ahoguen los ~5 tokens de texto.
    - SLIDING WINDOW: Alimenta a Gemma con una ventana deslizante de
      los últimos N TRs (texto + audio), preservando contexto narrativo.
    - SUB-MUESTREO 2 Hz: Extrae 3 sub-muestras por TR (cada 0.5s) y las
      promedia. Captura fonemas y microexpresiones que 1 frame/TR pierde.
"""

import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import librosa
import numpy as np
from PIL import Image

from src.config import ModelConfig
from src.temporal_alignment import TemporalPooling


class OfflineExtractor:
    """
    Extrae y guarda los hidden states pooled de Gemma 4 para cada estímulo.
    
    El resultado es un archivo .pt por cada ventana temporal (TR) que contiene
    un tensor de forma (1536,) — listo para alimentar directamente al Bottleneck.
    
    Mejora v2: Extracción Multi-Layer
        En lugar de usar solo la última capa (-1), extraemos de dos bloques
        de capas intermedias siguiendo el protocolo de TriBE v2:
        
        Gemma 4 E2B tiene 35 capas (num_hidden_layers).
        
        Bloque 1 (50%–75%):  capas 17–26  → promedio → (1, SeqLen, 1536)
        Bloque 2 (75%–100%): capas 26–35  → promedio → (1, SeqLen, 1536)
        
        Se promedian ambos bloques y luego se aplica temporal pooling.
        
        Esto preserva las representaciones semánticas intermedias que están
        mejor alineadas con la actividad cerebral humana.
    
    Args:
        config: ModelConfig con la configuración del modelo.
        pooling_mode: 'mean' o 'conv1d' — cómo colapsar la secuencia de tokens.
        multi_layer: Si True, extrae de múltiples capas (recomendado para brain encoding).
    """
    
    def __init__(
        self,
        config: ModelConfig = None,
        pooling_mode: str = "conv1d",
        multi_layer: bool = True,
    ):
        self.config = config or ModelConfig()
        self.pooling_mode = pooling_mode
        self.multi_layer = multi_layer
        
        device = self.config.device
        dtype = self.config.dtype
        
        print("╔══════════════════════════════════════════════╗")
        print("║   Offline Feature Extractor                 ║")
        print(f"║   Device: {str(device):<35s}║")
        print(f"║   Dtype:  {str(dtype):<35s}║")
        if multi_layer:
            print("║   Mode:   Multi-Layer (TriBE v2 style)      ║")
        else:
            print(f"║   Layer:  {str(self.config.extraction_layer):<35s}║")
        print("╚══════════════════════════════════════════════╝")
        
        # Cargar Gemma 4 para extracción
        self._load_gemma(device, dtype)
        
        # Calcular capas GLOBALES para multi-layer extraction.
        # Gemma 4 E2B tiene 35 capas con patrón 4:1 (4 sliding + 1 global).
        # hidden_states tiene 36 entradas (embedding + 35 capas).
        #
        # Las capas de atención LOCAL (sliding window 512 tokens) solo "ven"
        # ~7 segundos de contexto. Con nuestra ventana de 30s, solo las capas
        # GLOBALES integran todo el contexto narrativo.
        #
        # Capas globales (0-indexed): 4, 9, 14, 19, 24, 29, 34
        # En hidden_states (1-indexed): 5, 10, 15, 20, 25, 30, 35
        self.num_layers = 35  # hardcoded from Gemma 4 E2B config.json
        
        # Todas las capas globales (patrón: cada 5ta capa empezando desde 4)
        self.global_layer_indices = [i + 1 for i in range(4, self.num_layers, 5)]
        # → [5, 10, 15, 20, 25, 30, 35] en hidden_states
        
        if self.multi_layer:
            # Bloque 1: capas globales intermedias (50%-75% de profundidad)
            # Capas 19, 24 (hidden_states[20, 25])
            self.block1_indices = [i for i in self.global_layer_indices if 20 <= i <= 25]
            # Bloque 2: capas globales profundas (75%-100%)
            # Capas 29, 34 (hidden_states[30, 35])
            self.block2_indices = [i for i in self.global_layer_indices if 30 <= i <= 35]
            
            b1_layers = [i - 1 for i in self.block1_indices]  # 0-indexed para display
            b2_layers = [i - 1 for i in self.block2_indices]
            print(f"  🌐 Bloque 1 (Global Attn, mid): capas {b1_layers} ({len(self.block1_indices)} capas)")
            print(f"  🌐 Bloque 2 (Global Attn, deep): capas {b2_layers} ({len(self.block2_indices)} capas)")
            print(f"  ⚠️  Capas locales (sliding 512 tokens) EXCLUIDAS — no ven la ventana completa")
        
        # Temporal pooling
        self.temporal_pooling = TemporalPooling(
            hidden_size=self.config.gemma_hidden_size,
            mode=pooling_mode,
            dtype=dtype,
        ).to(device)
    
    def _load_gemma(self, device: torch.device, dtype: torch.dtype) -> None:
        """Carga Gemma 4 E2B-it en modo de solo inferencia."""
        from transformers import AutoProcessor, AutoModelForMultimodalLM
        
        print(f"  Descargando {self.config.model_id}...")
        
        self.processor = AutoProcessor.from_pretrained(self.config.model_id)
        
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
            device_map="auto" if device.type == "cuda" else None,
        )
        
        if device.type != "cuda":
            self.model = self.model.to(device)
        
        # Congelar y poner en modo eval
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  ✅ Gemma 4 cargado ({total_params:,} parámetros, modo eval)")
    
    def _pool_multi_layer(
        self,
        hidden_states: tuple,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Extrae y promedia hidden states de capas GLOBALES (TriBE v2 + Gemma-aware).
        Incluye LayerNorm pre-pooling para evitar el ahogamiento del texto.
        
        Protocolo:
            1. Extrae SOLO capas de atención global (no locales)
            2. Promedia las capas del Bloque 1 (globales mid: 19, 24)
            3. Promedia las capas del Bloque 2 (globales deep: 29, 34)
            4. Promedia ambos bloques → un tensor (1, SeqLen, 1536)
            5. LayerNorm por token → normaliza la escala de cada token
            6. Aplica temporal pooling → (1, 1536)
        
        ¿Por qué SOLO capas globales?
            Gemma 4 E2B usa un patrón 4:1 (4 capas sliding window de 512
            tokens + 1 capa de atención global completa). Las capas locales
            solo "ven" ~7 segundos de contexto (~512 tokens). Nuestra ventana
            deslizante inyecta ~30s de narrativa (~2000+ tokens). Si extraemos
            de capas locales, perdemos el contexto que tanto nos costó preservar.
            Las capas globales SÍ integran la secuencia completa.
        
        ¿Por qué LayerNorm pre-pooling?
            Gemma genera ~280 tokens de imagen, ~200+ de audio, pero solo
            ~5 de texto por TR. Sin normalización, al hacer mean(SeqLen),
            los tokens de imagen/audio "ahogan" a los de texto (~1% del total).
            LayerNorm normaliza cada token individualmente a media=0, std=1,
            igualando su peso numérico antes del promedio.
        
        Args:
            hidden_states: Tupla de (num_layers+1) tensores (1, SeqLen, 1536).
            attention_mask: (1, SeqLen) máscara de atención.
            
        Returns:
            pooled: (1536,) vector pooled multi-layer.
        """
        # Promediar capas globales del Bloque 1 (mid-depth: 19, 24)
        block1_states = [hidden_states[i] for i in self.block1_indices]
        block1_avg = torch.stack(block1_states, dim=0).mean(dim=0)  # (1, SeqLen, 1536)
        
        # Promediar capas globales del Bloque 2 (deep: 29, 34)
        block2_states = [hidden_states[i] for i in self.block2_indices]
        block2_avg = torch.stack(block2_states, dim=0).mean(dim=0)  # (1, SeqLen, 1536)
        
        # Promediar ambos bloques (peso igual)
        combined = (block1_avg + block2_avg) / 2.0  # (1, SeqLen, 1536)
        
        # LayerNorm PRE-POOLING: normalizar cada token antes de promediar.
        # Cada token (ya sea de imagen, audio o texto) queda con media≈0, std≈1.
        # Esto evita que los ~280 tokens de imagen ahoguen a los ~5 de texto.
        combined = F.layer_norm(combined, [combined.shape[-1]])
        
        # Temporal pooling: (1, SeqLen, 1536) → (1, 1536)
        pooled = self.temporal_pooling(combined, attention_mask)
        
        return pooled.squeeze(0).cpu()  # (1536,)
    
    @torch.no_grad()
    def extract_single(
        self,
        text: str = None,
        images: list = None,
        audio = None,
    ) -> torch.Tensor:
        """
        Extrae un vector pooled (1536,) de un estímulo multimodal.
        
        Si multi_layer=True, extrae de múltiples capas intermedias.
        Si multi_layer=False, extrae solo de la capa configurada.
        
        Args:
            text: Texto descriptivo o prompt.
            images: Lista de imágenes PIL o frames de video.
            audio: Audio array o path.
        
        Returns:
            pooled: Tensor de forma (1536,)
        """
        # Preparar inputs con el processor de Gemma
        processor_kwargs = {"return_tensors": "pt"}
        
        # Para modelos multimodales recientes de HuggingFace (Gemma 4),
        # es indispensable usar apply_chat_template para que reconozca los
        # tokens especiales de visión y audio y no los tome como texto crudo.
        content = []
        
        if images:
            for _ in images:
                content.append({"type": "image"})
            processor_kwargs["images"] = images
            
        if audio is not None:
            content.append({"type": "audio"})
            processor_kwargs["audio"] = audio
            
        # Siempre debe haber un texto si usamos AutoProcessor multimodal
        final_text = text if text else "Analyze this."
        content.append({"type": "text", "text": final_text})
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        processor_kwargs["text"] = prompt
        
        inputs = self.processor(**processor_kwargs)
        
        # Mover al dispositivo
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass — extraer TODOS los hidden states
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        
        attention_mask = inputs.get("attention_mask", None)
        
        if self.multi_layer:
            # Multi-layer extraction (TriBE v2 style)
            return self._pool_multi_layer(outputs.hidden_states, attention_mask)
        else:
            # Single layer extraction (legacy)
            hidden_states = outputs.hidden_states[self.config.extraction_layer]
            pooled = self.temporal_pooling(hidden_states, attention_mask)
            return pooled.squeeze(0).cpu()
    
    @torch.no_grad()
    def extract_batch(
        self,
        stimuli: list[dict],
        output_path: str,
    ) -> torch.Tensor:
        """
        Extrae features para múltiples estímulos y los guarda en disco.
        
        Cada estímulo es un dict con keys opcionales:
            {"text": ..., "images": ..., "audio": ...}
        
        Args:
            stimuli: Lista de dicts con los estímulos por TR.
            output_path: Path donde guardar el tensor de features.
            save_raw_hidden_states: Si True, también guarda los hidden_states
                                   sin poolear (para experimentar con diferentes
                                   estrategias de pooling después).
        
        Returns:
            features: Tensor (num_stimuli, 1536) con todos los features extraídos.
        """
        all_features = []
        
        for i, stim in enumerate(stimuli):
            pooled = self.extract_single(
                text=stim.get("text"),
                images=stim.get("images"),
                audio=stim.get("audio"),
            )
            all_features.append(pooled)
            
            if (i + 1) % 10 == 0 or i == len(stimuli) - 1:
                print(f"  Procesado {i + 1}/{len(stimuli)} estímulos")
        
        # Stack: lista de (1536,) → (num_stimuli, 1536)
        features = torch.stack(all_features, dim=0)
        
        # Guardar en disco
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(features, output_path)
        
        # Tamaño del archivo
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  💾 Features guardados en {output_path}")
        print(f"     Forma: {features.shape}")
        print(f"     Tamaño: {size_mb:.1f} MB")
        print(f"     Dtype: {features.dtype}")
        
        return features


def generate_real_extraction(
    dataset_dir: str,
    output_dir: str,
    tr_duration: float = 1.5,
    multi_layer: bool = True,
    context_window_trs: int = 20,
):
    """
    Lee los archivos reales, los sincroniza por TR, y extrae los features
    usando una sliding window para preservar contexto narrativo.
    
    Usa el archivo de Excel oficial como reloj maestro y se detiene cuando
    termina el video.
    
    Sliding Window:
        En vez de darle a Gemma solo 1.5s aislados por TR, le damos una
        ventana deslizante de los últimos N TRs (texto acumulado + audio
        acumulado). Esto permite que la atención global de Gemma contextualice
        la escena actual con la historia narrativa reciente.
        
        Con Gemma 4 E2B (128K tokens de contexto), podemos caber ventanas
        de hasta ~250 TRs (375 segundos) sin problemas.
    
    Args:
        dataset_dir: Directorio con video, audio y Excel.
        output_dir: Directorio donde guardar el tensor de features.
        tr_duration: Duración del TR en segundos (default: 1.5 para Sherlock).
        multi_layer: Si True, usa extracción multi-layer (TriBE v2 style).
        context_window_trs: Número de TRs de contexto narrativo (default: 20 = 30s).
    """
    import pandas as pd
    
    print("=" * 60)
    print("🎬 Iniciando Extracción Real Multimodal (Guiado por Excel)")
    print("=" * 60)
    
    # 1. Cargar el Reloj Maestro (Excel)
    excel_path = Path(dataset_dir) / "Sherlock_Segments_1000_NN_2017.xlsx"
    if not excel_path.exists():
        print(f"❌ Error: No se encontró la 'Biblia' de sincronización en {excel_path}")
        return
        
    df = pd.read_excel(excel_path)
    print(f"  ✅ Reloj maestro cargado: {len(df)} segmentos semánticos.")
    
    # 2. Cargar Video para determinar la duración real
    video_path = f"{dataset_dir}/stimuli_Sherlock.m4v"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_sec = total_frames / fps if fps > 0 else 0
    num_video_trs = int(video_duration_sec / tr_duration)
    
    print(f"  ✅ Video abierto. FPS: {fps:.2f} | Duración: {video_duration_sec:.1f}s")
    print(f"  🎯 TRs a extraer (limitado por el video): {num_video_trs}")
    
    # 3. Cargar Audio Original (directo del video)
    print("  ⏳ Obteniendo audio original del video...")
    try:
        audio_waveform, sr = librosa.load(video_path, sr=16000)
        print(f"  ✅ Audio original cargado directo del video. Sample Rate: {sr}")
    except Exception:
        import subprocess
        import tempfile
        tmp_audio = tempfile.mktemp(suffix=".wav")
        print("  🎬 Extrayendo pistas de audio con ffmpeg...")
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", tmp_audio],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        audio_waveform, sr = librosa.load(tmp_audio, sr=16000)
        os.remove(tmp_audio)
        print(f"  ✅ Audio original cargado vía ffmpeg. Sample Rate: {sr}")
    
    # 4. Inicializar el Extractor
    extractor = OfflineExtractor(multi_layer=multi_layer)
    all_features = []
    error_count = 0
    
    # Pre-calcular todas las descripciones de texto para facilitar la sliding window
    all_texts = []
    for tr_idx in range(num_video_trs):
        tr_number = tr_idx + 1
        match = df[(df["Start Time (TRs, 1.5s)"] <= tr_number) & (df["End Time (TRs, 1.5s)"] >= tr_number)]
        if not match.empty:
            all_texts.append(str(match.iloc[0]["Scene Details - A Level"]))
        else:
            all_texts.append("")
    
    print(f"  🪟 Sliding Window: {context_window_trs} TRs ({context_window_trs * tr_duration:.0f}s de contexto narrativo)")
    
    # Frecuencia de sub-muestreo: 2 Hz (2 muestras/segundo)
    # Para un TR de 1.5s → 3 sub-muestras por TR (a 0.25s, 0.75s, 1.25s)
    # TriBE v2 (OpenReview) demostró que 2Hz captura transitorios rápidos
    # (fonemas, microexpresiones) que mejoran significativamente la predicción.
    sub_sample_rate_hz = 2.0
    sub_samples_per_tr = max(1, int(tr_duration * sub_sample_rate_hz))
    sub_interval = tr_duration / sub_samples_per_tr
    print(f"  🔬 Sub-muestreo: {sub_sample_rate_hz} Hz → {sub_samples_per_tr} sub-muestras/TR (cada {sub_interval:.2f}s)")
    
    # 5. Bucle de extracción sincronizada TR por TR (con Sliding Window + 2Hz)
    for tr_idx in range(num_video_trs):
        current_time_sec = tr_idx * tr_duration
        
        # === SLIDING WINDOW: Construir contexto narrativo acumulativo ===
        window_start = max(0, tr_idx - context_window_trs + 1)
        
        # --- A. TEXTO CON CONTEXTO NARRATIVO ---
        context_texts = []
        prev_text = ""
        for w_idx in range(window_start, tr_idx + 1):
            t = all_texts[w_idx]
            if t and t != prev_text:
                context_texts.append(t)
                prev_text = t
        
        if context_texts:
            text_prompt = " ".join(context_texts)
        else:
            text_prompt = "No description available."
            
        # --- B. AUDIO CON CONTEXTO DE LA VENTANA ---
        window_start_sec = window_start * tr_duration
        window_end_sec = current_time_sec + tr_duration
        
        audio_start_sample = int(window_start_sec * sr)
        audio_end_sample = int(window_end_sec * sr)
        
        if audio_end_sample <= len(audio_waveform):
            audio_segment = audio_waveform[audio_start_sample:audio_end_sample]
        else:
            if audio_start_sample < len(audio_waveform):
                audio_segment = audio_waveform[audio_start_sample:]
            else:
                audio_segment = np.zeros(int(tr_duration * sr))
        
        # --- C. SUB-MUESTREO 2 Hz: Extraer múltiples frames por TR ---
        # En vez de 1 frame en el punto medio, extraemos N frames equiespaciados.
        # Cada sub-muestra pasa por Gemma con el MISMO texto/audio de contexto
        # pero con un frame DIFERENTE (captura micro-expresiones y transitorios).
        sub_features = []
        
        for sub_idx in range(sub_samples_per_tr):
            # Frame en el centro de cada sub-intervalo
            sub_center = current_time_sec + (sub_idx + 0.5) * sub_interval
            frame_idx = int(sub_center * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_idx, total_frames - 1))
            ret, frame = cap.read()
            
            images_list = []
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                images_list.append(pil_img)
            
            try:
                pooled_state = extractor.extract_single(
                    text=text_prompt,
                    images=images_list if images_list else None,
                    audio=audio_segment
                )
                
                if pooled_state.isnan().any():
                    pooled_state = torch.zeros(1536)
                    error_count += 1
                
                sub_features.append(pooled_state)
            except Exception as e:
                print(f"Error en TR {tr_idx} sub {sub_idx} (Seg {sub_center:.1f}): {e}")
                sub_features.append(torch.zeros(1536))
                error_count += 1
        
        # Promediar las sub-muestras → 1 vector (1536,) por TR
        tr_feature = torch.stack(sub_features, dim=0).mean(dim=0)
        all_features.append(tr_feature)
            
        if (tr_idx + 1) % 50 == 0:
            w_size = min(tr_idx + 1, context_window_trs)
            print(f"  ⏳ Procesados {tr_idx + 1}/{num_video_trs} TRs... (ventana: {w_size} TRs, {sub_samples_per_tr} sub/TR)")

    cap.release()
    
    # 6. Guardar el tensor maestro
    if len(all_features) > 0:
        final_tensor = torch.stack(all_features, dim=0) # (num_video_trs, 1536)
        
        output_path = Path(output_dir) / "real_stimulus_features.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(final_tensor, output_path)
        
        # Reporte de calidad
        nan_rows = final_tensor.isnan().any(dim=1).sum().item()
        zero_rows = (final_tensor.abs().sum(dim=1) == 0).sum().item()
        
        print(f"\n🎉 ¡Extracción finalizada! Guardado en: {output_path}")
        print(f"  Forma del tensor: {final_tensor.shape}")
        print(f"  Dtype: {final_tensor.dtype}")
        print(f"  Errores durante extracción: {error_count}")
        print(f"  Filas NaN: {nan_rows} | Filas all-zero (fallback): {zero_rows}")
        
        if multi_layer:
            print("  🧠 Modo: Multi-Layer (Global Attention Only)")
            print("  📐 LayerNorm pre-pooling: Activado (anti-ahogamiento de texto)")
            print(f"  🌐 Capas extraídas: {[i-1 for i in extractor.block1_indices + extractor.block2_indices]} (solo globales)")
        else:
            print(f"  🧠 Modo: Single-Layer ({extractor.config.extraction_layer})")
        print(f"  🪟 Sliding Window: {context_window_trs} TRs ({context_window_trs * tr_duration:.0f}s)")
        print(f"  🔬 Sub-muestreo: 2 Hz ({sub_samples_per_tr} sub-muestras/TR)")
    else:
        print("\n❌ No se extrajeron features.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extractor offline de features de Gemma 4"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/features",
        help="Directorio de salida (default: ./data/features).",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./dataset",
        help="Directorio donde están los archivos del dataset real (video, audio, text npy).",
    )
    parser.add_argument(
        "--tr_duration",
        type=float,
        default=1.5,
        help="Duración del TR de la fMRI (default: 1.5s).",
    )
    parser.add_argument(
        "--single_layer",
        action="store_true",
        help="Si se especifica, usa solo la última capa en vez de multi-layer.",
    )
    parser.add_argument(
        "--context_window",
        type=int,
        default=20,
        help="TRs de contexto narrativo en la sliding window (default: 20 = 30s).",
    )
    
    args = parser.parse_args()
    
    generate_real_extraction(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        tr_duration=args.tr_duration,
        multi_layer=not args.single_layer,
        context_window_trs=args.context_window,
    )
