"""
================================================================================
extract_features_v2.py — Codificador Narrativo con Gemma 4 para Algonauts 2025
================================================================================

PROPÓSITO GENERAL
-----------------
Versión V2 del extractor: "Codificador Narrativo".
Para cada TR[t], Gemma 4 recibe un contexto rico y acumulativo:
  • TEXTO:     Últimas 1024 palabras del transcript (narrativa acumulada)
  • AUDIO:     30 segundos previos de audio a 16kHz
  • IMÁGENES:  24 frames equiespaciados de los últimos ~32 segundos

Cuadrícula de extracción: 1 Hz (1 muestra por TR de 1.49s).

USO
---
    # Extracción piloto narrativa (24 chunks)
    python -m src.extract_features_v2 --pilot

    # Solo texto (baseline)
    python -m src.extract_features_v2 --pilot --text_only

    # Fusionar
    python -m src.extract_features_v2 --merge

AUTOR
-----
Proyecto GemmaBe — Brain Encoding con Gemma 4 E2B-it.
================================================================================
"""

import argparse
import io
import json
import time
from datetime import datetime
from pathlib import Path

import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"      # Silencia warnings de OpenCV
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"     # Silencia ffmpeg/swscaler dentro de OpenCV
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Reduce fragmentación VRAM

import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import h5py
from transformers import AutoProcessor, AutoModelForMultimodalLM

from src.config import ModelConfig
from src.temporal_alignment import TemporalPooling
from src.prepare_fmri import get_fmri_paths, load_algonauts_fmri
from src.utils import extract_audio_from_mkv


# =============================================================================
# OfflineExtractor — Motor de extracción con Gemma 4 congelado
# =============================================================================

class OfflineExtractor:
    """
    Carga Gemma 4 E2B-it en modo evaluación (congelado) y extrae embeddings
    de 1536 dimensiones para estímulos multimodales.

    Args:
        config:       ModelConfig con device, dtype, y parámetros de Gemma 4.
        pooling_mode: "mean" (promedio temporal).
        multi_layer:  True → extrae y promedia capas globales 20, 25, 30, 35.
                      False → extrae solo la última capa (35).

    Atributos:
        model:        Gemma4ForConditionalGeneration cargado en device.
        processor:    AutoProcessor que tokeniza texto, procesa imágenes/audio.
        temporal_pooling: nn.Module que colapsa (B, SeqLen, 1536) → (B, 1536).
        _hooks:       Lista de forward hooks registrados en capas objetivo.
        _hook_outputs: Dict que acumula outputs de hooks durante forward.
    """

    def __init__(self, config: ModelConfig = None, multi_layer: bool = True):
        self.config = config or ModelConfig()
        self.multi_layer = multi_layer
        device = self.config.device
        dtype = self.config.dtype

        print("╔══════════════════════════════════════════════╗")
        print("║   Offline Feature Extractor (Algonauts)     ║")
        print(f"║   Device: {str(device):<35s}║")
        print(f"║   Dtype:  {str(dtype):<35s}║")
        mode_str = "Multi-Layer (TriBE v2)" if multi_layer else str(self.config.extraction_layer)
        print(f"║   Mode:   {mode_str:<35s}║")

        self._load_gemma(device, dtype)

        # Índices de capas con atención GLOBAL en Gemma 4 (patrón 4 sliding + 1 global)
        self.num_layers = 35
        self.global_layer_indices = [i + 1 for i in range(4, self.num_layers, 5)]
        if self.multi_layer:
            self.block1_indices = [i for i in self.global_layer_indices if 20 <= i <= 25]
            self.block2_indices = [i for i in self.global_layer_indices if 30 <= i <= 35]

        self.temporal_pooling = TemporalPooling(
            hidden_size=self.config.gemma_hidden_size, dtype=dtype,
        ).to(device)

        # Hooks: capturan hidden states de capas específicas sin output_hidden_states=True
        self._hooks = []
        self._hook_outputs = {}
        self._register_hooks()

    def _load_gemma(self, device: torch.device, dtype: torch.dtype) -> None:
        """Carga Gemma 4 E2B-it desde HuggingFace en el device especificado."""
        print(f"  Cargando {self.config.model_id}...")
        self.processor = AutoProcessor.from_pretrained(self.config.model_id)

        load_kwargs = {"device_map": "auto", "torch_dtype": dtype}
        if device.type != "cuda":
            load_kwargs.pop("device_map")

        self.model = AutoModelForMultimodalLM.from_pretrained(self.config.model_id, **load_kwargs)

        if device.type != "cuda":
            self.model = self.model.to(device)

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  ✅ Gemma 4 cargado ({total_params:,} parámetros)")

    def _get_transformer_layers(self):
        """Navega la jerarquía anidada de Gemma4 para encontrar el ModuleList de capas."""
        m = self.model
        if hasattr(m, 'model'):
            m = m.model
        if hasattr(m, 'language_model'):
            m = m.language_model
        if hasattr(m, 'model'):
            m = m.model
        if not hasattr(m, 'layers'):
            raise AttributeError("No se encontraron transformer layers en el modelo")
        return m.layers

    def _make_hook(self, layer_idx):
        """Factory que crea un hook que captura el hidden state de una capa."""
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self._hook_outputs[layer_idx] = hidden
        return hook

    def _register_hooks(self):
        """Registra forward hooks SOLO en las capas que realmente necesitamos extraer."""
        if self.multi_layer:
            target_layers = set(self.block1_indices + self.block2_indices)
        else:
            extraction = self.config.extraction_layer
            if extraction < 0:
                extraction = self.num_layers
            target_layers = {extraction}

        layers = self._get_transformer_layers()
        for layer_idx in target_layers:
            module = layers[layer_idx - 1]  # 0-indexed
            h = module.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(h)

    def _pool_multi_layer(self, hidden_states, attention_mask=None):
        """Promedia capas globales de dos bloques (20-25 y 30-35), normaliza, y hace pooling."""
        b1 = torch.stack([hidden_states[i] for i in self.block1_indices]).mean(0)
        b2 = torch.stack([hidden_states[i] for i in self.block2_indices]).mean(0)
        combined = (b1 + b2) / 2.0
        m = combined.mean(dim=-1, keepdim=True)
        s = combined.std(dim=-1, keepdim=True) + 1e-6
        return self.temporal_pooling((combined - m) / s, attention_mask)

    def _pool_single_layer(self, hidden_states, attention_mask=None):
        """Hace pooling sobre la capa única especificada en config.extraction_layer."""
        return self.temporal_pooling(hidden_states[self.config.extraction_layer], attention_mask)

    def _run_forward_and_pool(self, inputs: dict):
        """
        Ejecuta un forward de Gemma 4 usando los hooks registrados (sin output_hidden_states).

        Args:
            inputs: Diccionario con input_ids, pixel_values, audio_values, etc.

        Returns:
            Tensor (B, 1536) con el feature vector pooled.
        """
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.config.dtype)
        if "audio_values" in inputs:
            inputs["audio_values"] = inputs["audio_values"].to(self.config.dtype)

        self._hook_outputs.clear()
        outputs = self.model(**inputs, output_hidden_states=False)

        # Reconstruir lista de hidden states desde los hooks
        hs_list = [None] * (self.num_layers + 1)
        for idx, tensor in self._hook_outputs.items():
            hs_list[idx] = tensor

        if self.multi_layer:
            pooled = self._pool_multi_layer(hs_list, inputs.get("attention_mask"))
        else:
            pooled = self._pool_single_layer(hs_list, inputs.get("attention_mask"))

        # Limpieza explícita para evitar acumulación en MPS
        del inputs, outputs, hs_list
        self._hook_outputs.clear()
        return pooled

    @torch.inference_mode()
    def extract_single(self, text: str, images: list = None, audio: np.ndarray = None) -> torch.Tensor:
        """
        Extrae un vector (1536,) para un instante temporal dado.

        Incluye fallback automático por OOM: si 64 imágenes exceden la VRAM,
        reintenta con 32 imágenes. Si aún falla, devuelve ceros.

        Args:
            text:   Prompt de texto (acumulado de la ventana de contexto).
            images: Lista de PIL.Image (frames del video). None = omitir.
            audio:  Array numpy de audio (16kHz). None = omitir.

        Returns:
            Tensor (1536,) en CPU.
        """
        try:
            return self._extract_single_impl(text, images, audio)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            oom_str = str(e).lower()
            is_oom = "out of memory" in oom_str or "cuda" in oom_str
            if not is_oom:
                raise
            if images and len(images) > 32:
                print(f"    OOM con {len(images)} imgs, reintentando con 32...")
                torch.cuda.empty_cache()
                return self._extract_single_impl(text, images[-32:], audio)
            else:
                print("    OOM incluso con ≤32 imgs, devolviendo ceros.")
                torch.cuda.empty_cache()
                return torch.zeros(self.config.gemma_hidden_size)

    def _extract_single_impl(self, text: str, images: list = None, audio: np.ndarray = None) -> torch.Tensor:
        """Implementación real de extract_single (sin manejo de OOM)."""
        user_content = []
        if images:
            for img in images:
                user_content.append({"type": "image", "image": img})
        if audio is not None and len(audio) > 100:
            user_content.append({"type": "audio", "audio": audio})
        user_content.append({"type": "text", "text": text or "Describe this scene."})

        messages = [{"role": "user", "content": user_content}]
        formatted_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        proc_kwargs = {"text": formatted_text, "return_tensors": "pt"}
        if images:
            proc_kwargs["images"] = images
        if audio is not None and len(audio) > 100:
            proc_kwargs["audio"] = audio
            proc_kwargs["sampling_rate"] = 16000

        inputs = self.processor(**proc_kwargs)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        pooled = self._run_forward_and_pool(inputs)
        return pooled.squeeze(0).cpu()


# =============================================================================
# Pre-extracción de frames — Elimina seeks aleatorios de OpenCV
# =============================================================================

def _preextract_frames_sequential(mkv_path: Path, needed_indices: set, fps: float, total_frames: int):
    """
    Extrae frames específicos de un video .mkv mediante un ÚNICO scan secuencial.

    Esta función lee el video de principio a fin una 
    sola vez y guarda solo los frames necesarios.

    Args:
        mkv_path:       Ruta al archivo de video.
        needed_indices: Set de índices de frame que se necesitan.
        fps:            Frames por segundo del video.
        total_frames:   Número total de frames.

    Returns:
        Dict {frame_index: bytes_jpeg} con los frames comprimidos en memoria.
    """
    if not needed_indices:
        return {}

    cap = cv2.VideoCapture(str(mkv_path))
    needed_set = set(needed_indices)
    frames = {}
    current = 0
    max_needed = max(needed_indices)

    while current <= max_needed:
        ret, frame = cap.read()
        if not ret:
            break
        if current in needed_set:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=95)
            frames[current] = buf.getvalue()
        current += 1

    cap.release()
    return frames


# =============================================================================
# process_chunk — Extracción completa de un chunk de video
# =============================================================================

# ─── Constantes del Codificador Narrativo ───
NARRATIVE_MAX_WORDS = 1024       # Últimas 1024 palabras del transcript
NARRATIVE_AUDIO_SEC = 30.0       # 30 segundos de audio previo
NARRATIVE_NUM_FRAMES = 32        # 32 frames de contexto visual (~32s, 1 cada ~1.0s)
NARRATIVE_SUB_SAMPLES = 1        # 1 Hz (1 muestra por TR, sin promedio)


def process_chunk(
    extractor: OfflineExtractor,
    mkv_path: Path,
    tsv_path: Path,
    num_trs_expected: int,
    tr_duration: float,
    text_only: bool = False,
) -> torch.Tensor:
    """
    Codificador Narrativo: extrae features para un chunk completo.

    Para cada TR[t], Gemma 4 recibe:
      • Últimas 1024 palabras del transcript (texto acumulado)
      • 30 segundos de audio previo (16kHz)
      • 64 frames equiespaciados de los últimos ~32 segundos

    Cuadrícula: 2 Hz → 2 muestras por TR, luego se promedian.

    Args:
        extractor:        OfflineExtractor con Gemma 4 cargado.
        mkv_path:         Ruta al video .mkv.
        tsv_path:         Ruta al transcript .tsv alineado a TRs.
        num_trs_expected: Número de TRs que debería tener este chunk.
        tr_duration:      Duración de cada TR en segundos (1.49).
        text_only:        Si True, omite imágenes y audio.

    Returns:
        Tensor (num_trs_expected, 1536) con un feature vector por TR.
    """
    sr = 16000

    # 1. Cargar texto por TR desde el transcript
    df = pd.read_csv(tsv_path, sep='\t')
    if len(df) != num_trs_expected:
        print(f"  TSV TRs ({len(df)}) != HDF5 TRs ({num_trs_expected})")
        num_trs_expected = min(len(df), num_trs_expected)

    # 2. Pre-construir texto acumulado (lista de todas las palabras hasta cada TR)
    all_words: list[list[str]] = []
    cumulative: list[str] = []
    for i in range(num_trs_expected):
        t = str(df.iloc[i]['text_per_tr'])
        if t != "nan" and t.strip() not in ("", "[]"):
            cumulative.extend(t.strip().split())
        all_words.append(list(cumulative))  # snapshot

    # 3. Cargar audio completo del video (16kHz)
    print(f"  Extrayendo audio de {mkv_path.name}...")
    audio_waveform = extract_audio_from_mkv(mkv_path, sr=sr)
    if len(audio_waveform) <= 1:
        print("  Audio vacío, usando silencio.")
        audio_waveform = np.zeros(int(num_trs_expected * tr_duration * sr))

    # 4. Abrir video para obtener metadata (fps, total_frames)
    cap = cv2.VideoCapture(str(mkv_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Usamos ~32s de ventana visual (cobertura similar al audio de 30s)
    VISUAL_WINDOW_SEC = 32.0

    needed_frame_indices = set()
    if not text_only:
        for tr_idx in range(num_trs_expected):
            current_time_sec = (tr_idx + 1) * tr_duration
            window_start = max(0.0, current_time_sec - VISUAL_WINDOW_SEC)
            for fi in range(NARRATIVE_NUM_FRAMES):
                t_sec = window_start + (fi + 0.5) * (current_time_sec - window_start) / NARRATIVE_NUM_FRAMES
                fidx = min(int(t_sec * fps), total_frames - 1)
                needed_frame_indices.add(max(0, fidx))

    # 6. Pre-extraer frames secuencialmente una sola vez
    if text_only:
        frame_cache = {}
        print("  📝 Modo TEXT-ONLY: omitiendo video y audio.")
    else:
        print(f"  🎞️  Pre-extrayendo {len(needed_frame_indices)} frames únicos...")
        frame_cache = _preextract_frames_sequential(mkv_path, needed_frame_indices, fps, total_frames)

    def _get_frame(fidx: int):
        """Recupera un frame del caché como PIL.Image."""
        raw = frame_cache.get(fidx)
        if raw is None:
            return None
        return Image.open(io.BytesIO(raw))

    # 7. Función interna: prepara text, images, audio para un sub-sample
    def _prepare_inputs(tr_idx: int, sub_offset_sec: float):
        current_time_sec = tr_idx * tr_duration + sub_offset_sec

        # TEXTO: últimas 1024 palabras acumuladas hasta este TR
        words = all_words[tr_idx]
        text_prompt = " ".join(words[-NARRATIVE_MAX_WORDS:]) if words else ""

        if text_only:
            return text_prompt, None, None

        # AUDIO: últimos 30 segundos hasta current_time_sec
        a_end = int(current_time_sec * sr)
        a_start = int(max(0, current_time_sec - NARRATIVE_AUDIO_SEC) * sr)
        a_end = min(a_end, len(audio_waveform))
        if a_start < a_end:
            audio_segment = audio_waveform[a_start:a_end]
        else:
            audio_segment = np.zeros(int(tr_duration * sr))

        # IMÁGENES: 64 frames equiespaciados en los últimos ~32 segundos
        window_start = max(0.0, current_time_sec - VISUAL_WINDOW_SEC)
        images_for_gemma = []
        for fi in range(NARRATIVE_NUM_FRAMES):
            t_sec = window_start + (fi + 0.5) * (current_time_sec - window_start) / NARRATIVE_NUM_FRAMES
            fidx = min(int(t_sec * fps), total_frames - 1)
            fidx = max(0, fidx)
            img = _get_frame(fidx)
            if img is not None:
                images_for_gemma.append(img)

        return text_prompt, images_for_gemma if images_for_gemma else None, audio_segment

    # 8. Loop principal: 2 Hz (2 sub-muestras por TR, luego se promedian)
    error_count = 0
    chunk_features = []
    sub_interval = tr_duration / NARRATIVE_SUB_SAMPLES

    print(f"  Procesando {mkv_path.name} ({num_trs_expected} TRs × {NARRATIVE_SUB_SAMPLES} sub-muestras)...")

    for tr_idx in tqdm(range(num_trs_expected), desc="TRs", leave=False):
        sub_features = []
        for sub_idx in range(NARRATIVE_SUB_SAMPLES):
            sub_offset = (sub_idx + 0.5) * sub_interval
            text_prompt, images, audio_segment = _prepare_inputs(tr_idx, sub_offset)
            try:
                pooled = extractor.extract_single(
                    text=text_prompt, images=images, audio=audio_segment,
                )
                if pooled.isnan().any():
                    pooled = torch.zeros(1536)
                    error_count += 1
                sub_features.append(pooled)
            except Exception as e:
                if error_count < 5:
                    print(f"\n  Error en TR {tr_idx}.{sub_idx}: {e}")
                sub_features.append(torch.zeros(1536))
                error_count += 1

        # Promedio de las 2 sub-muestras → 1 vector por TR
        chunk_features.append(torch.stack(sub_features).mean(dim=0))

    if error_count > 0:
        print(f"   {error_count} errores de extracción en este chunk.")
    return torch.stack(chunk_features, dim=0)


# =============================================================================
# Gestión de chunks y subset piloto
# =============================================================================

def get_chunks_info(algonauts_dir: str) -> list[dict]:
    """
    Lista ordenada de todos los chunks del dataset Algonauts 2025.

    Para cada chunk, retorna un dict con:
        task:   "friends" o "movie10"
        key:    key del HDF5 (ej: "ses-001_task-s01e02a")
        mkv:    Path al archivo de video
        tsv:    Path al transcript
        num_trs: Número de TRs en el chunk

    Fuente de verdad: los archivos HDF5 de fMRI de sub-01.
    """
    alg_dir = Path(algonauts_dir)
    paths = get_fmri_paths(alg_dir, "sub-01")
    all_chunks_info = []
    for task, h5_path in paths.items():
        if not h5_path.exists():
            continue
        _, keys = load_algonauts_fmri(h5_path, include_test=False)
        with h5py.File(h5_path, 'r') as f:
            for key in keys:
                chunk_len = len(f[key][:])
                if task == "friends":
                    chunk_id = key.split("task-")[1]
                    season_num = int(chunk_id[1:3])
                    mkv = alg_dir / f"stimuli/movies/friends/s{season_num}/friends_{chunk_id}.mkv"
                    tsv = alg_dir / f"stimuli/transcripts/friends/s{season_num}/friends_{chunk_id}.tsv"
                else:
                    task_part = key.split("task-")[1]
                    chunk_id = task_part.split("_")[0]
                    movie_name = "".join([c for c in chunk_id if not c.isdigit()])
                    mkv = alg_dir / f"stimuli/movies/movie10/{movie_name}/{chunk_id}.mkv"
                    tsv = alg_dir / f"stimuli/transcripts/movie10/{movie_name}/movie10_{chunk_id}.tsv"
                all_chunks_info.append({"task": task, "key": key, "mkv": mkv, "tsv": tsv, "num_trs": chunk_len})
    return all_chunks_info


def filter_pilot_indices(all_chunks: list[dict]) -> set:
    """
    Selecciona un subset piloto representativo de ~24 chunks:
      • Friends: temporadas 1, 3, 5, 6 — primer episodio de cada una (partes a+b).
        → 4 temporadas × 1 episodio × 2 partes = 8 chunks
      • movie10: primeros 4 chunks de cada una de las 4 películas.
        → 4 películas × 4 chunks = 16 chunks
      Total: ~24 chunks.

    Los chunks excluidos por TriBE v2 (s05e20a, s04e01a, etc.) ya son filtrados
    por load_algonauts_fmri, por lo que no aparecen en all_chunks.

    Retorna un set con los índices ORIGINALES en all_chunks que pertenecen al piloto.
    """
    pilot_indices = set()

    # --- Friends: temporadas 1, 3, 5, 6 — solo el primer episodio de cada una ---
    friends = [(i, c) for i, c in enumerate(all_chunks) if c["task"] == "friends"]
    for season in [1, 3, 5, 6]:
        season_eps = {}
        for idx, c in friends:
            key = c["key"]
            if f"task-s{season:02d}" not in key:
                continue
            chunk_id = key.split("task-")[1]
            if len(chunk_id) >= 6 and chunk_id[3] == 'e':
                try:
                    ep_num = int(chunk_id[4:6])
                except ValueError:
                    continue
                season_eps.setdefault(ep_num, []).append(idx)

        # Solo el primer episodio de la temporada (partes a y b incluidas)
        for ep_num in sorted(season_eps.keys())[:1]:
            pilot_indices.update(season_eps[ep_num])

    # --- movie10: primeros 4 chunks de cada película ---
    # Agrupa por nombre de película (bourne, wolf, etc.)
    movie_groups: dict[str, list[int]] = {}
    for i, c in enumerate(all_chunks):
        if c["task"] != "movie10":
            continue
        key = c["key"]
        # El nombre de la película está en la key.
        # Formatos posibles: "ses-001_task-bourne01" o "ses-006_task-life01_run-1"
        task_part = key.split("task-")[1]             # "bourne01" | "life01_run-1"
        task_part = task_part.split("_run-")[0]       # quita sufijo "_run-N" si existe
        movie_name = "".join(ch for ch in task_part if not ch.isdigit())  # "bourne" | "life"
        movie_groups.setdefault(movie_name, []).append(i)

    for movie_name, indices in movie_groups.items():
        # Tomar los primeros 4 chunks en orden (ya están ordenados por el HDF5)
        pilot_indices.update(sorted(indices)[:4])

    return pilot_indices



# =============================================================================
# Tracker JSON — Resume de extracción sin pérdida de progreso
# =============================================================================

def load_or_create_chunk_tracker(output_dir: Path, chunks_info: list[dict]) -> dict:
    """
    Carga o inicializa el JSON de tracking de chunks procesados.

    El tracker persiste en `output_dir/processed_chunks.json` y permite:
      • Reanudar la extracción después de un crash o reinicio.
      • Saber exactamente qué chunks faltan sin escanear el disco.
      • Filtrar el fMRI para incluir solo los chunks procesados.

    Si el JSON existe pero el número de chunks no coincide con chunks_info,
    se asume que el dataset cambió y se reinicia el tracker.
    """
    tracker_path = output_dir / "processed_chunks.json"
    if tracker_path.exists():
        with open(tracker_path, 'r') as f:
            tracker = json.load(f)
        if len(tracker.get("chunks", [])) == len(chunks_info):
            return tracker
        print("   Tracker JSON desactualizado (número de chunks cambió). Reiniciando...")

    tracker = {
        "created_at": datetime.now().isoformat(),
        "total_chunks": len(chunks_info),
        "processed_count": 0,
        "chunks": [
            {
                "index": i,
                "key": c["key"],
                "task": c["task"],
                "num_trs": c["num_trs"],
                "num_trs_extracted": None,
                "processed": False,
                "timestamp": None,
                "error": None,
            }
            for i, c in enumerate(chunks_info)
        ],
    }
    return tracker


def save_chunk_tracker(output_dir: Path, tracker: dict) -> None:
    """Guarda el estado actual del tracker en `output_dir/processed_chunks.json`."""
    tracker_path = output_dir / "processed_chunks.json"
    with open(tracker_path, 'w') as f:
        json.dump(tracker, f, indent=2, ensure_ascii=False)


def mark_chunk_processed(tracker: dict, chunk_index: int, error: str = None, num_trs_extracted: int = None) -> None:
    """Marca un chunk como procesado (o con error) y actualiza contadores."""
    entry = tracker["chunks"][chunk_index]
    entry["processed"] = error is None
    entry["timestamp"] = datetime.now().isoformat() if error is None else entry["timestamp"]
    entry["error"] = error
    if num_trs_extracted is not None:
        entry["num_trs_extracted"] = num_trs_extracted
    tracker["processed_count"] = sum(1 for c in tracker["chunks"] if c["processed"])
    tracker["last_updated"] = datetime.now().isoformat()


# =============================================================================
# merge_chunks — Concatena chunk_XXX.pt en un solo tensor
# =============================================================================

def merge_chunks(output_dir: str, algonauts_dir: str, tracker: dict = None):
    """
    Concatena todos los archivos `chunk_XXX.pt` en `real_stimulus_features.pt`.

    Si se provee un tracker, SOLO fusiona los chunks marcados como `processed=True`.
    Esto permite generar un tensor consistente incluso si solo se extrajo un subset.

    Args:
        output_dir:  Directorio base de features (contiene subdirectorio `chunks/`).
        algonauts_dir: Directorio del dataset (para obtener la lista completa de chunks).
        tracker:     (opcional) Dict del tracker JSON. Si None, fusiona todos los chunks.
    """
    out_dir = Path(output_dir)
    chunks_dir = out_dir / "chunks"
    chunks_info = get_chunks_info(algonauts_dir)

    if tracker is not None:
        indices_to_merge = [i for i, c in enumerate(tracker["chunks"]) if c["processed"]]
        expected_trs = sum(
            tracker["chunks"][i].get("num_trs_extracted") or tracker["chunks"][i]["num_trs"]
            for i in indices_to_merge
        )
        print(f"\n🔗 Fusionando {len(indices_to_merge)}/{len(chunks_info)} chunks procesados...")
    else:
        indices_to_merge = list(range(len(chunks_info)))
        expected_trs = sum(c["num_trs"] for c in chunks_info)
        print(f"\n🔗 Fusionando {len(indices_to_merge)} chunks...")

    all_tensors, missing = [], []
    for i in indices_to_merge:
        p = chunks_dir / f"chunk_{i:03d}.pt"
        if not p.exists():
            missing.append(i)
            continue
        all_tensors.append(torch.load(p, weights_only=True))

    if missing:
        print(f"  ❌ Faltan {len(missing)} archivos .pt: {missing[:20]}")
        return

    if not all_tensors:
        print("  ❌ No hay tensores para fusionar.")
        return

    final = torch.cat(all_tensors, dim=0)
    out_path = out_dir / "real_stimulus_features.pt"
    torch.save(final, out_path)
    print(f"  ✅ {out_path} → {final.shape}")
    if final.shape[0] == expected_trs:
        print("  🎉 ¡Sincronización perfecta!")
    else:
        print(f"  ⚠️ Diferencia: {abs(final.shape[0] - expected_trs)} TRs")
    return final


# =============================================================================
# generate_real_extraction — Orquestador principal
# =============================================================================

def generate_real_extraction(
    algonauts_dir, output_dir, tr_duration,
    multi_layer,
    pilot=False, text_only=False,
    match_tracker=None,
):
    """
    Orquesta la extracción de features para todo el dataset (o el subset piloto).

    Carga Gemma 4 una sola vez, itera sobre los chunks, y guarda cada resultado
    junto con el tracker JSON para permitir reanudación.

    Args:
        algonauts_dir:  Directorio raíz de Algonauts 2025.
        output_dir:     Directorio donde se guardan features y tracker.
        tr_duration:    Duración de cada TR en segundos.
        multi_layer:    True → extrae múltiples capas; False → solo última.
        pilot:          True → procesa solo el subset piloto (~24 chunks).
        text_only:      True → omite imágenes y audio, solo transcript.
        match_tracker:  Path a tracker v1. Si se provee, usa EXACTAMENTE los
                        mismos índices que v1 para permitir comparativa directa.
    """
    out_dir = Path(output_dir)
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    print("\n Determinando chunks desde prepare_fmri.py...")
    all_chunks = get_chunks_info(algonauts_dir)
    total_trs = sum(c["num_trs"] for c in all_chunks)
    print(f"{len(all_chunks)} chunks totales en el dataset, {total_trs:,} TRs.")

    # ─── Calcular índices del subset (match_tracker tiene prioridad) ─────────
    if match_tracker is not None:
        import json as _json
        with open(match_tracker, 'r') as f:
            v1_tracker = _json.load(f)
        pilot_indices = {
            c["index"] for c in v1_tracker["chunks"] if c["processed"]
        }
        pilot_trs = sum(all_chunks[i]["num_trs"] for i in pilot_indices if i < len(all_chunks))
        print(f"MATCH TRACKER v1: {len(pilot_indices)} chunks alineados, {pilot_trs:,} TRs.")
    elif pilot:
        pilot_indices = filter_pilot_indices(all_chunks)
        pilot_trs = sum(all_chunks[i]["num_trs"] for i in pilot_indices)
        print(f"MODO PILOTO: {len(pilot_indices)} chunks seleccionados, {pilot_trs:,} TRs.")
        print("   Friends: temporadas 1, 3, 5, 6 (primeros 3 episodios c/u)")
        print("   movie10: las 4 películas completas")
    else:
        pilot_indices = set(range(len(all_chunks)))

    # Cargar o crear tracker JSON (con TODOS los chunks para índices estables)
    tracker = load_or_create_chunk_tracker(out_dir, all_chunks)
    processed_before = tracker["processed_count"]

    # Detectar chunks ya existentes en disco (solo dentro del subset actual)
    disk_done = {
        i for i in pilot_indices
        if (chunks_dir / f"chunk_{i:03d}.pt").exists()
    }
    for i in disk_done:
        if not tracker["chunks"][i]["processed"]:
            mark_chunk_processed(tracker, i)
    if tracker["processed_count"] > processed_before:
        save_chunk_tracker(out_dir, tracker)
        print(f"Se detectaron {tracker['processed_count'] - processed_before} chunks ya existentes en disco.")

    pending = sum(1 for i in pilot_indices if not tracker["chunks"][i]["processed"])
    if tracker["processed_count"] > 0:
        print(f"RESUME: {tracker['processed_count']} hechos en total, {pending} pendientes en este subset.")
    if pending == 0:
        print("Todo procesado. Usa --merge.")
        save_chunk_tracker(out_dir, tracker)
        return

    available_indices = {i for i in pilot_indices if all_chunks[i]["mkv"].exists() and all_chunks[i]["tsv"].exists()}
    skipped = pilot_indices - available_indices
    if skipped:
        print(f"⚠️  {len(skipped)} chunks sin estímulos descargados (se saltan)")
    pilot_indices = available_indices
    if not pilot_indices:
        print("❌ Ningún chunk tiene estímulos disponibles. Descarga con datalad.")
        return

    config = ModelConfig()
    extractor = OfflineExtractor(config=config, multi_layer=multi_layer)

    start_time = time.time()
    done_count = 0
    for i, chunk in enumerate(all_chunks):
        if i not in pilot_indices:
            continue
        if tracker["chunks"][i]["processed"]:
            continue
        print(f"\n[{i+1}/{len(all_chunks)}] {chunk['key']} ({chunk['num_trs']} TRs)")
        try:
            features = process_chunk(
                extractor=extractor, mkv_path=chunk["mkv"], tsv_path=chunk["tsv"],
                num_trs_expected=chunk["num_trs"], tr_duration=tr_duration,
                text_only=text_only,
            )

            # ─── Validación de integridad ────────────────────────────
            chunk_path = chunks_dir / f"chunk_{i:03d}.pt"
            num_trs = features.shape[0]
            has_nan = torch.isnan(features).any().item()
            has_inf = torch.isinf(features).any().item()
            zero_trs = (features.abs().sum(dim=-1) == 0).sum().item()
            valid = not has_nan and not has_inf and zero_trs < num_trs

            if not valid:
                print(f"  ⚠️  CHUNK CORRUPTO: NaN={has_nan} Inf={has_inf} zeros={zero_trs}/{num_trs}")
                print("  ⚠️  Guardando de todas formas pero marcando error.")
                torch.save(features, chunk_path)
                mark_chunk_processed(tracker, i, error=f"corrupto: nan={has_nan} inf={has_inf} zeros={zero_trs}")
            else:
                torch.save(features, chunk_path)
                # Re-leer para verificar integridad en disco
                verify = torch.load(chunk_path, weights_only=True)
                assert verify.shape == features.shape, f"Shape mismatch: {verify.shape} vs {features.shape}"
                del verify
                mark_chunk_processed(tracker, i, num_trs_extracted=num_trs)
                if zero_trs > 0:
                    print(f"  ⚠️  {zero_trs}/{num_trs} TRs con embeddings vacíos (primeros TRs sin contexto visual)")
                print(f"  ✅ chunk_{i:03d}.pt verificado ({num_trs} TRs, {features.shape[1]}d)")

            save_chunk_tracker(out_dir, tracker)
            done_count += 1
            elapsed = time.time() - start_time
            eta = (pending - done_count) * elapsed / done_count / 60 if done_count > 0 else 0
            print(f"     ETA restante: {eta:.0f} min ({done_count}/{pending} chunks)")
        except Exception as e:
            print(f"Error CRÍTICO en chunk {i}: {e}")
            mark_chunk_processed(tracker, i, error=str(e))
            save_chunk_tracker(out_dir, tracker)

    print("\nTodos los chunks disponibles han sido procesados. Fusionando...")
    merge_chunks(output_dir, algonauts_dir, tracker=tracker)


# =============================================================================
# Entry point — CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Codificador Narrativo v2: Gemma 4 × Algonauts 2025"
    )
    parser.add_argument("--output_dir", type=str, default="./data/features_v2",
                        help="Directorio de salida para features y tracker JSON")
    parser.add_argument("--algonauts_dir", type=str, default="./algonauts_2025",
                        help="Directorio raíz del dataset Algonauts 2025")
    parser.add_argument("--tr_duration", type=float, default=1.49,
                        help="Duración de cada TR en segundos (default 1.49)")
    parser.add_argument("--single_layer", action="store_true",
                        help="Extraer solo la última capa en vez de multi-layer")
    parser.add_argument("--pilot", action="store_true",
                        help="Modo piloto: extrae solo 24 chunks representativos")
    parser.add_argument("--text_only", action="store_true",
                        help="Modo text-only: omite imágenes y audio, solo transcript")
    parser.add_argument("--merge", action="store_true",
                        help="Solo fusionar chunks ya procesados en real_stimulus_features.pt")
    parser.add_argument("--status", action="store_true",
                        help="Mostrar progreso de extracción desde el tracker JSON")
    parser.add_argument("--match_tracker", type=str, default=None,
                        help="Path a tracker v1. Alinea v2 exactamente con los mismos chunks que v1.")

    args = parser.parse_args()

    if args.merge:
        out_dir = Path(args.output_dir)
        tracker = None
        tracker_path = out_dir / "processed_chunks.json"
        if tracker_path.exists():
            with open(tracker_path, 'r') as f:
                tracker = json.load(f)
            print(f"📋 Tracker cargado: {tracker['processed_count']}/{tracker['total_chunks']} chunks.")
        merge_chunks(args.output_dir, args.algonauts_dir, tracker=tracker)

    elif args.status:
        ci = get_chunks_info(args.algonauts_dir)
        out_dir = Path(args.output_dir)
        tracker_path = out_dir / "processed_chunks.json"
        if tracker_path.exists():
            with open(tracker_path, 'r') as f:
                tracker = json.load(f)
            total_trs = sum(
                c.get("num_trs_extracted") or c["num_trs"]
                for c in tracker["chunks"] if c["processed"]
            )
            print(f"📊 {tracker['processed_count']}/{tracker['total_chunks']} chunks procesados")
            print(f"   TRs cubiertos: {total_trs:,}")
            mismatched = [
                c for c in tracker["chunks"]
                if c["processed"] and c.get("num_trs_extracted") and c["num_trs_extracted"] != c["num_trs"]
            ]
            if mismatched:
                print(f"{len(mismatched)} chunks con truncación (TSV vs HDF5)")
        else:
            cd = out_dir / "chunks"
            done = sum(1 for i in range(len(ci)) if (cd / f"chunk_{i:03d}.pt").exists())
            print(f"{done}/{len(ci)} chunks procesados (sin tracker JSON)")

    else:
        print("╔══════════════════════════════════════════════╗")
        print("║   CODIFICADOR NARRATIVO v2                  ║")
        print(f"║   Texto:    últimas {NARRATIVE_MAX_WORDS} palabras             ║")
        print(f"║   Audio:    {NARRATIVE_AUDIO_SEC:.0f}s previos                     ║")
        print(f"║   Frames:   {NARRATIVE_NUM_FRAMES} equiespaciados (~32s)        ║")
        print(f"║   Grid:     {NARRATIVE_SUB_SAMPLES} Hz (promediado por TR)       ║")
        print("╚══════════════════════════════════════════════╝")
        generate_real_extraction(
            args.algonauts_dir, args.output_dir, args.tr_duration,
            not args.single_layer,
            pilot=args.pilot, text_only=args.text_only,
            match_tracker=args.match_tracker,
        )
