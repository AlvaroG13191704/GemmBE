"""
extract_features_v3.py — Extracción ViT + Conformer + Texto (Pre-Proyección)

Nueva estrategia alineada con TriBE:
  • VISUAL:    Salida del ViT (SigLIP) ANTES del multi_modal_projector
               → captura características de bajo nivel (bordes, texturas, movimiento)
  • AUDITIVO:  Salida del Conformer ANTES del audio_multi_modal_projector
               → captura pitch, fonemas, envolvente espectral
  • TEXTO:     Hidden states del decoder (capas 20-35) igual que v2
               → captura semántica narrativa de alto nivel

Guarda 3 tensores SEPARADOS por chunk (no concatenados), de modo que
el modelo downstream puede aprender a pesar cada modalidad independientemente:
  data/features_v3/chunks/
    chunk_000_vit.pt        (N_TRs, D_vit)
    chunk_000_conformer.pt  (N_TRs, D_conformer)
    chunk_000_text.pt       (N_TRs, 1536)

Uso:
    python -m src.extract_features_v3
    python -m src.extract_features_v3 --text_only
    python -m src.extract_features_v3 --merge
    python -m src.extract_features_v3 --status
    python -m src.extract_features_v3 --check_missing
"""

import argparse
import io
import json
import time
from datetime import datetime
from pathlib import Path

import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import h5py
from transformers import AutoProcessor, AutoModelForMultimodalLM, AutoConfig

# pyrefly: ignore [missing-import]
from src.config import ModelConfig
# pyrefly: ignore [missing-import]
from src.utils.temporal_alignment import TemporalPooling
# pyrefly: ignore [missing-import]
from src.utils.prepare_fmri import get_fmri_paths, load_algonauts_fmri
# pyrefly: ignore [missing-import]
from src.utils.helpers import extract_audio_from_mkv


# =============================================================================
# Constantes
# =============================================================================
NARRATIVE_MAX_WORDS   = 1024
NARRATIVE_AUDIO_SEC   = 30.0
NARRATIVE_NUM_FRAMES  = 32
NARRATIVE_SUB_SAMPLES = 1


# =============================================================================
# Utilidades de descubrimiento de módulos
# =============================================================================

def _find_vit_last_layer(model: nn.Module):
    """
    Encuentra la última capa del ViT de Gemma 4 E2B.

    Confirmado por inspect_gemma4_modules.py:
      Ruta exacta: model.vision_tower.encoder.layers  (ModuleList, 16 capas)
      Última capa: model.vision_tower.encoder.layers[15]  Gemma4VisionEncoderLayer
      Output dim: vision_config.hidden_size = 768

    NOTA: Hookeamos la PENULTIMA capa del ViT (antes del pooler y embed_vision),
    no el embed_vision.embedding_projection (eso ya es el projector al LLM space).
    """
    # Ruta primaria confirmada (Gemma 4 E2B-it)
    primary_paths = [
        "model.vision_tower.encoder.layers",   # google/gemma-4-E2B-it
        "vision_tower.encoder.layers",          # por si el nivel superior cambia
        "model.vision_model.encoder.layers",    # fallback SigLIP standalone
        "vision_model.encoder.layers",
    ]
    for path in primary_paths:
        parts = path.split(".")
        obj = model
        try:
            for p in parts:
                obj = getattr(obj, p)
            last_layer = obj[-1]   # último elemento del ModuleList
            print(f"  [ViT] ✓ Capa encontrada: {path}[-1]  ({type(last_layer).__name__})")
            return last_layer, path
        except AttributeError:
            continue

    print("  [ViT] ADVERTENCIA: No se encontró la capa ViT. Se usarán ceros.")
    return None, None


def _find_conformer_last_layer(model: nn.Module):
    """
    Encuentra el punto de corte del Audio Conformer de Gemma 4 E2B.

    Confirmado por inspect_gemma4_modules.py:
      Ruta de capas:   model.audio_tower.layers  (ModuleList, 12 capas)
      Output proj:     model.audio_tower.output_proj  (Linear, dim=1024)
      Projector LLM:   model.embed_audio.embedding_projection  (Linear)

    ESTRATEGIA: Hookeamos model.audio_tower.output_proj (Linear), que es
    el ULTIMO modulo del audio_tower ANTES de que embed_audio lo proyecte
    al espacio del LLM. Su output tiene dim = audio_config.hidden_size = 1024.
    Esto es equivalente a los features de Wav2Vec2 pre-proyeccion en TriBE.
    """
    # Intentar primero el output_proj (mas limpio, ya es un vector por frame)
    output_proj_paths = [
        "model.audio_tower.output_proj",   # Gemma 4 E2B-it (confirmado)
        "audio_tower.output_proj",
    ]
    for path in output_proj_paths:
        parts = path.split(".")
        obj = model
        try:
            for p in parts:
                obj = getattr(obj, p)
            # Es un Linear (no un ModuleList), retornarlo directamente
            print(f"  [Conformer] ✓ output_proj: {path}  ({type(obj).__name__})")
            return obj, path
        except AttributeError:
            continue

    # Fallback: ultima capa del ModuleList de Conformer
    layer_paths = [
        "model.audio_tower.layers",
        "audio_tower.layers",
        "model.audio_model.conformer_layers",
    ]
    for path in layer_paths:
        parts = path.split(".")
        obj = model
        try:
            for p in parts:
                obj = getattr(obj, p)
            last_layer = obj[-1]
            print(f"  [Conformer] Fallback: {path}[-1]  ({type(last_layer).__name__})")
            return last_layer, path
        except AttributeError:
            continue

    print("  [Conformer] ADVERTENCIA: No se encontró. Se usarán ceros.")
    return None, None


def _find_vit_model(model: nn.Module):
    """Retorna el sub-modelo ViT completo (para forward separado)."""
    for attr in ["model.vision_tower", "vision_tower", "vision_model", "image_encoder"]:
        try:
            obj = model
            for p in attr.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    return None


def _find_audio_model(model: nn.Module):
    """Retorna el sub-modelo audio completo (para forward separado)."""
    for attr in ["model.audio_tower", "audio_tower", "audio_model", "audio_encoder"]:
        try:
            obj = model
            for p in attr.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    return None


# =============================================================================
# OfflineExtractorV3 — Motor de extracción con 3 ramas separadas
# =============================================================================

class OfflineExtractorV3:
    """
    Extractor que produce 3 vectores independientes por TR:
      • vit_emb:       (D_vit,)      — salida pre-proyección del ViT
      • conformer_emb: (D_conformer,)— salida pre-proyección del Conformer
      • text_emb:      (1536,)       — hidden states del decoder LLM

    Estos vectores NO están concatenados. El modelo downstream (TriBEStyleModel)
    aprende a pesarlos con FFNs independientes antes de fusionarlos.
    """

    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        device = self.config.device
        dtype  = self.config.dtype

        print("=" * 55)
        print("Feature Extractor V3 — ViT + Conformer + Text")
        print(f"  Device: {device}")
        print(f"  Dtype:  {dtype}")
        print("=" * 55)

        self._load_gemma(device, dtype)
        self._setup_hooks()

        self.temporal_pooling = TemporalPooling(
            hidden_size=self.config.gemma_hidden_size, dtype=dtype,
        ).to(device)

    def _load_gemma(self, device, dtype):
        print(f"  Cargando {self.config.model_id}...")
        self.processor = AutoProcessor.from_pretrained(self.config.model_id)

        load_kwargs = {"device_map": "auto", "torch_dtype": dtype}
        if device.type != "cuda":
            load_kwargs.pop("device_map")

        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.config.model_id, **load_kwargs
        )
        if device.type != "cuda":
            self.model = self.model.to(device)

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  Gemma 4 cargado ({total:,} params)")

    def _setup_hooks(self):
        """Registra hooks en ViT, Conformer y decoder por separado."""
        self._hook_outputs = {}
        self._hooks = []

        # ── Hook 1: ViT (última capa del SigLIP encoder) ──────────────────
        print("\nBuscando módulos para hooks:")
        vit_layer, vit_path = _find_vit_last_layer(self.model)
        self._vit_available = vit_layer is not None
        if vit_layer is not None:
            h = vit_layer.register_forward_hook(self._make_hook("vit_last"))
            self._hooks.append(h)

        # ── Hook 2: Conformer (última capa del audio encoder) ─────────────
        conf_layer, conf_path = _find_conformer_last_layer(self.model)
        self._conformer_available = conf_layer is not None
        if conf_layer is not None:
            h = conf_layer.register_forward_hook(self._make_hook("conformer_last"))
            self._hooks.append(h)

        # ── Hook 3: Decoder LLM (igual que v2) ────────────────────────────
        self.num_layers = 35
        text_layer_indices = [i + 1 for i in range(4, self.num_layers, 5)]
        self._block1_indices = [i for i in text_layer_indices if 20 <= i <= 25]
        self._block2_indices = [i for i in text_layer_indices if 30 <= i <= 35]
        target_layers = set(self._block1_indices + self._block2_indices)

        layers = self._get_transformer_layers()
        for idx in target_layers:
            h = layers[idx - 1].register_forward_hook(self._make_hook(f"text_{idx}"))
            self._hooks.append(h)

        print(f"  [Texto] Hooks en capas: {sorted(target_layers)}")
        print(f"  Total hooks registrados: {len(self._hooks)}")

    def _make_hook(self, key: str):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self._hook_outputs[key] = hidden
        return hook

    def _get_transformer_layers(self):
        m = self.model
        if hasattr(m, "model"):
            m = m.model
        if hasattr(m, "language_model"):
            m = m.language_model
        if hasattr(m, "model"):
            m = m.model
        return m.layers

    # ─────────────────────────────────────────────────────────────────────────
    # Extracción de cada rama
    # ─────────────────────────────────────────────────────────────────────────

    def _pool_vit(self) -> torch.Tensor:
        """
        Recoge la salida del hook de la última capa ViT y hace mean pooling.

        Gemma4VisionEncoderLayer output shape:
          La capa retorna una tupla: (hidden_states, ...) donde
          hidden_states es (B, num_patches, 768).

        Hacemos mean sobre los patches (dim=1) → (768,) por frame.
        Si hay múltiples frames, el hook se dispara varias veces y capturamos
        el último. El mean temporal sobre frames lo hace la lógica de extraccion.
        """
        hidden = self._hook_outputs.get("vit_last")
        if hidden is None:
            return torch.zeros(self.config.vit_hidden_size)   # dim=768
        # hidden puede ser tupla (output, attn_weights) según la impl de HF
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        # Shape esperado: (B, num_patches, 768) → mean sobre patches
        if hidden.dim() == 3:
            return hidden.squeeze(0).mean(dim=0).float().cpu()   # (768,)
        elif hidden.dim() == 2:
            return hidden.mean(dim=0).float().cpu()              # ya era (patches, 768)
        else:
            return hidden.float().cpu().flatten()[:self.config.vit_hidden_size]

    def _pool_conformer(self) -> torch.Tensor:
        """
        Recoge la salida del hook del output_proj del Audio Conformer.

        model.audio_tower.output_proj es un Linear que proyecta los
        features del Conformer. Su output es (B, T_audio, 1024).
        Hacemos mean sobre T (tiempo) → (1024,) por TR.
        """
        hidden = self._hook_outputs.get("conformer_last")
        if hidden is None:
            return torch.zeros(self.config.conformer_hidden_size)  # dim=1024
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        # Shape esperado: (B, T_audio_frames, 1024)
        if hidden.dim() == 3:
            return hidden.squeeze(0).mean(dim=0).float().cpu()    # (1024,)
        elif hidden.dim() == 2:
            return hidden.mean(dim=0).float().cpu()               # (T, 1024) → mean
        else:
            return hidden.float().cpu().flatten()[:self.config.conformer_hidden_size]

    def _pool_text(self, attention_mask=None) -> torch.Tensor:
        """
        Igual que v2: promedio de capas 20-25 y 30-35 del decoder,
        luego temporal pooling.
        """
        hs_list = [None] * (self.num_layers + 1)
        for key, tensor in self._hook_outputs.items():
            if key.startswith("text_"):
                idx = int(key.split("_")[1])
                hs_list[idx] = tensor

        b1 = torch.stack([hs_list[i] for i in self._block1_indices]).mean(0)
        b2 = torch.stack([hs_list[i] for i in self._block2_indices]).mean(0)
        combined = (b1 + b2) / 2.0
        m = combined.mean(dim=-1, keepdim=True)
        s = combined.std(dim=-1, keepdim=True) + 1e-6
        pooled = self.temporal_pooling((combined - m) / s, attention_mask)
        return pooled.squeeze(0).float().cpu()

    # ─────────────────────────────────────────────────────────────────────────
    # Forward completo (un TR)
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_single_impl(
        self,
        text: str,
        images: list = None,
        audio: np.ndarray = None,
        text_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retorna (vit_emb, conformer_emb, text_emb) para un TR.
        En text_only, vit y conformer son ceros.
        """
        # Construir contenido del mensaje
        user_content = []
        if not text_only:
            if images:
                for img in images:
                    user_content.append({"type": "image", "image": img})
            if audio is not None and len(audio) > 100:
                user_content.append({"type": "audio", "audio": audio})
        user_content.append({"type": "text", "text": text or "Describe this scene."})

        if "-it" in self.config.model_id:
            messages = [{"role": "user", "content": user_content}]
            formatted = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Modelo base (sin -it): formateamos manualmente el prompt con marcadores
            text_part = text or "Describe this scene."
            media_prefix = ""
            if not text_only:
                if images:
                    media_prefix += "<image>" * len(images)
                if audio is not None and len(audio) > 100:
                    media_prefix += "<audio>"
            formatted = f"{media_prefix} {text_part}"


        proc_kwargs = {"text": formatted, "return_tensors": "pt"}
        if not text_only:
            if images:
                proc_kwargs["images"] = images
            if audio is not None and len(audio) > 100:
                proc_kwargs["audio"] = audio
                proc_kwargs["sampling_rate"] = 16000

        inputs = self.processor(**proc_kwargs)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.config.dtype)
        if "audio_values" in inputs:
            inputs["audio_values"] = inputs["audio_values"].to(self.config.dtype)

        # Forward pass — todos los hooks se disparan aquí
        self._hook_outputs.clear()
        self.model(**inputs, output_hidden_states=False)

        # Recoger cada rama
        if text_only:
            vit_emb  = torch.zeros(self.config.vit_hidden_size)
            conf_emb = torch.zeros(self.config.conformer_hidden_size)
        else:
            vit_emb  = self._pool_vit()
            conf_emb = self._pool_conformer()

        text_emb = self._pool_text(inputs.get("attention_mask"))
        self._hook_outputs.clear()

        return vit_emb, conf_emb, text_emb

    @torch.inference_mode()
    def extract_single(
        self,
        text: str,
        images: list = None,
        audio: np.ndarray = None,
        text_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extrae (vit_emb, conformer_emb, text_emb) con manejo de OOM.
        """
        try:
            return self._extract_single_impl(text, images, audio, text_only)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() and "cuda" not in str(e).lower():
                raise
            # Reintentar con menos frames
            if images and len(images) > 16:
                print(f"    OOM con {len(images)} imgs, reintentando con 16...")
                torch.cuda.empty_cache()
                return self._extract_single_impl(text, images[-16:], audio, text_only)
            else:
                print(f"    OOM irrecuperable, retornando ceros.")
                torch.cuda.empty_cache()
                return (
                    torch.zeros(self.config.vit_hidden_size),
                    torch.zeros(self.config.conformer_hidden_size),
                    torch.zeros(self.config.gemma_hidden_size),
                )


# =============================================================================
# Pre-extracción de frames (idéntica a v2)
# =============================================================================

def _preextract_frames_sequential(mkv_path, needed_indices, fps, total_frames):
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
# process_chunk — Extrae (vit, conformer, text) para todos los TRs de un chunk
# =============================================================================

def process_chunk(
    extractor: OfflineExtractorV3,
    mkv_path: Path,
    tsv_path: Path,
    num_trs_expected: int,
    tr_duration: float,
    text_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Procesa un chunk y retorna 3 tensores:
        vit_features:       (N_TRs, D_vit)
        conformer_features: (N_TRs, D_conformer)
        text_features:      (N_TRs, 1536)
    """
    sr = 16000

    # 1. Texto por TR
    df = pd.read_csv(tsv_path, sep="\t")
    if len(df) != num_trs_expected:
        print(f"  WARNING: TSV TRs ({len(df)}) != HDF5 TRs ({num_trs_expected})")
        num_trs_expected = min(len(df), num_trs_expected)

    all_words = []
    cumulative = []
    for i in range(num_trs_expected):
        t = str(df.iloc[i]["text_per_tr"])
        if t != "nan" and t.strip() not in ("", "[]"):
            cumulative.extend(t.strip().split())
        all_words.append(list(cumulative))

    # 2. Audio
    print(f"  Extrayendo audio de {mkv_path.name}...")
    audio_waveform = extract_audio_from_mkv(mkv_path, sr=sr)
    if len(audio_waveform) <= 1:
        print("  WARNING: Audio vacío, usando silencio.")
        audio_waveform = np.zeros(int(num_trs_expected * tr_duration * sr))

    # 3. Video metadata
    cap = cv2.VideoCapture(str(mkv_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    VISUAL_WINDOW_SEC = 32.0

    # 4. Pre-calcular frames necesarios
    needed_frame_indices = set()
    if not text_only:
        for tr_idx in range(num_trs_expected):
            current_time_sec = (tr_idx + 1) * tr_duration
            window_start = max(0.0, current_time_sec - VISUAL_WINDOW_SEC)
            for fi in range(NARRATIVE_NUM_FRAMES):
                t_sec = window_start + (fi + 0.5) * (current_time_sec - window_start) / NARRATIVE_NUM_FRAMES
                fidx = min(int(t_sec * fps), total_frames - 1)
                needed_frame_indices.add(max(0, fidx))

    # 5. Pre-extraer frames
    if text_only:
        frame_cache = {}
        print("  TEXT-ONLY: saltando video y audio.")
    else:
        print(f"  Pre-extrayendo {len(needed_frame_indices)} frames únicos...")
        frame_cache = _preextract_frames_sequential(
            mkv_path, needed_frame_indices, fps, total_frames
        )

    def _get_frame(fidx):
        raw = frame_cache.get(fidx)
        return Image.open(io.BytesIO(raw)) if raw is not None else None

    # 6. Loop principal por TR
    error_count = 0
    vit_list, conf_list, text_list = [], [], []

    print(f"  Procesando {mkv_path.name} ({num_trs_expected} TRs)...")

    for tr_idx in tqdm(range(num_trs_expected), desc="TRs", leave=False):
        current_time_sec = (tr_idx + 1) * tr_duration
        words = all_words[tr_idx]
        text_prompt = " ".join(words[-NARRATIVE_MAX_WORDS:]) if words else ""

        if text_only:
            images, audio_segment = None, None
        else:
            a_end   = int(current_time_sec * sr)
            a_start = int(max(0, current_time_sec - NARRATIVE_AUDIO_SEC) * sr)
            a_end   = min(a_end, len(audio_waveform))
            audio_segment = (
                audio_waveform[a_start:a_end]
                if a_start < a_end
                else np.zeros(int(tr_duration * sr))
            )
            window_start = max(0.0, current_time_sec - VISUAL_WINDOW_SEC)
            images_list  = []
            for fi in range(NARRATIVE_NUM_FRAMES):
                t_sec = window_start + (fi + 0.5) * (current_time_sec - window_start) / NARRATIVE_NUM_FRAMES
                fidx  = max(0, min(int(t_sec * fps), total_frames - 1))
                img   = _get_frame(fidx)
                if img is not None:
                    images_list.append(img)
            images = images_list if images_list else None

        try:
            vit_emb, conf_emb, text_emb = extractor.extract_single(
                text=text_prompt, images=images, audio=audio_segment,
                text_only=text_only,
            )
            # Sanity check
            if vit_emb.isnan().any() or conf_emb.isnan().any() or text_emb.isnan().any():
                vit_emb   = torch.zeros(extractor.config.vit_hidden_size)
                conf_emb  = torch.zeros(extractor.config.conformer_hidden_size)
                text_emb  = torch.zeros(extractor.config.gemma_hidden_size)
                error_count += 1

            vit_list.append(vit_emb)
            conf_list.append(conf_emb)
            text_list.append(text_emb)

        except Exception as e:
            if error_count < 5:
                print(f"\n  ERROR en TR {tr_idx}: {e}")
            vit_list.append(torch.zeros(extractor.config.vit_hidden_size))
            conf_list.append(torch.zeros(extractor.config.conformer_hidden_size))
            text_list.append(torch.zeros(extractor.config.gemma_hidden_size))
            error_count += 1

    if error_count > 0:
        print(f"  WARNING: {error_count} errores de extracción.")

    return (
        torch.stack(vit_list,  dim=0),
        torch.stack(conf_list, dim=0),
        torch.stack(text_list, dim=0),
    )


# =============================================================================
# Tracker JSON (idéntico al de v2)
# =============================================================================

def get_chunks_info(algonauts_dir: str) -> list[dict]:
    alg_dir = Path(algonauts_dir)
    paths = get_fmri_paths(alg_dir, "sub-01")
    all_chunks_info = []
    for task, h5_path in paths.items():
        if not h5_path.exists():
            continue
        _, keys = load_algonauts_fmri(h5_path, include_test=False)
        with h5py.File(h5_path, "r") as f:
            for key in keys:
                chunk_len = len(f[key][:])
                if task == "friends":
                    chunk_id  = key.split("task-")[1]
                    season    = int(chunk_id[1:3])
                    mkv = alg_dir / f"stimuli/movies/friends/s{season}/friends_{chunk_id}.mkv"
                    tsv = alg_dir / f"stimuli/transcripts/friends/s{season}/friends_{chunk_id}.tsv"
                else:
                    task_part  = key.split("task-")[1]
                    chunk_id   = task_part.split("_")[0]
                    movie_name = "".join([c for c in chunk_id if not c.isdigit()])
                    mkv = alg_dir / f"stimuli/movies/movie10/{movie_name}/{chunk_id}.mkv"
                    tsv = alg_dir / f"stimuli/transcripts/movie10/{movie_name}/movie10_{chunk_id}.tsv"
                all_chunks_info.append({
                    "task": task, "key": key, "mkv": mkv,
                    "tsv": tsv, "num_trs": chunk_len,
                })
    return all_chunks_info


def load_or_create_tracker(output_dir: Path, chunks_info: list[dict]) -> dict:
    tracker_path = output_dir / "processed_chunks_v3.json"
    if tracker_path.exists():
        with open(tracker_path, "r") as f:
            tracker = json.load(f)
        if len(tracker.get("chunks", [])) == len(chunks_info):
            return tracker
        print("  WARNING: Tracker desactualizado. Restarting...")

    return {
        "created_at": datetime.now().isoformat(),
        "version": "v3",
        "total_chunks": len(chunks_info),
        "processed_count": 0,
        "chunks": [
            {
                "index": i, "key": c["key"], "task": c["task"],
                "num_trs": c["num_trs"], "num_trs_extracted": None,
                "processed": False, "timestamp": None, "error": None,
            }
            for i, c in enumerate(chunks_info)
        ],
    }


def save_tracker(output_dir: Path, tracker: dict):
    p = output_dir / "processed_chunks_v3.json"
    with open(p, "w") as f:
        json.dump(tracker, f, indent=2, ensure_ascii=False)


def mark_chunk_processed(tracker, i, error=None, num_trs=None):
    entry = tracker["chunks"][i]
    entry["processed"] = error is None
    entry["timestamp"] = datetime.now().isoformat() if error is None else entry["timestamp"]
    entry["error"]     = error
    if num_trs is not None:
        entry["num_trs_extracted"] = num_trs
    tracker["processed_count"] = sum(1 for c in tracker["chunks"] if c["processed"])
    tracker["last_updated"]    = datetime.now().isoformat()


# =============================================================================
# merge_chunks — Fusiona los 3 tensores separados en archivos finales
# =============================================================================

def merge_chunks(output_dir: str, algonauts_dir: str, tracker: dict = None):
    out_dir    = Path(output_dir)
    chunks_dir = out_dir / "chunks"
    chunks_info = get_chunks_info(algonauts_dir)

    if tracker is not None:
        indices = [i for i, c in enumerate(tracker["chunks"]) if c["processed"]]
    else:
        indices = list(range(len(chunks_info)))

    print(f"\nFusionando {len(indices)}/{len(chunks_info)} chunks (3 modalidades)...")

    vit_all, conf_all, text_all = [], [], []
    missing = []

    for i in indices:
        vit_p  = chunks_dir / f"chunk_{i:03d}_vit.pt"
        conf_p = chunks_dir / f"chunk_{i:03d}_conformer.pt"
        text_p = chunks_dir / f"chunk_{i:03d}_text.pt"

        if not (vit_p.exists() and conf_p.exists() and text_p.exists()):
            missing.append(i)
            continue

        vit_all.append(torch.load(vit_p,  weights_only=True))
        conf_all.append(torch.load(conf_p, weights_only=True))
        text_all.append(torch.load(text_p, weights_only=True))

    if missing:
        print(f"  WARNING: {len(missing)} chunks con archivos faltantes: {missing[:10]}")
        return

    final_vit  = torch.cat(vit_all,  dim=0)
    final_conf = torch.cat(conf_all, dim=0)
    final_text = torch.cat(text_all, dim=0)

    torch.save(final_vit,  out_dir / "real_vit_features.pt")
    torch.save(final_conf, out_dir / "real_conformer_features.pt")
    torch.save(final_text, out_dir / "real_text_features.pt")

    print(f"  ViT:       {final_vit.shape}  → real_vit_features.pt")
    print(f"  Conformer: {final_conf.shape} → real_conformer_features.pt")
    print(f"  Texto:     {final_text.shape} → real_text_features.pt")

    N = final_vit.shape[0]
    if final_conf.shape[0] == N == final_text.shape[0]:
        print(f"  ✅ Sync perfecto: {N} TRs en las 3 modalidades.")
    else:
        print(f"  ⚠️  Inconsistencia de TRs: vit={final_vit.shape[0]}, "
              f"conf={final_conf.shape[0]}, text={final_text.shape[0]}")


# =============================================================================
# Orquestador principal
# =============================================================================

def generate_extraction(
    algonauts_dir, output_dir, tr_duration,
    text_only=False, resume=True,
):
    out_dir    = Path(output_dir)
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    print("\nEscaneando dataset...")
    all_chunks  = get_chunks_info(algonauts_dir)
    total_trs   = sum(c["num_trs"] for c in all_chunks)
    print(f"  {len(all_chunks)} chunks totales, {total_trs:,} TRs.")

    tracker = load_or_create_tracker(out_dir, all_chunks)

    # Verificar estímulos disponibles
    available, missing_stim = set(), []
    for i, chunk in enumerate(all_chunks):
        if chunk["mkv"].exists() and chunk["tsv"].exists():
            available.add(i)
        else:
            missing_stim.append(i)

    if missing_stim:
        print(f"\n  ⚠️  {len(missing_stim)} chunks sin estímulos (serán omitidos).")

    # Detectar chunks ya en disco
    disk_done = {
        i for i in available
        if all([
            (chunks_dir / f"chunk_{i:03d}_vit.pt").exists(),
            (chunks_dir / f"chunk_{i:03d}_conformer.pt").exists(),
            (chunks_dir / f"chunk_{i:03d}_text.pt").exists(),
        ])
    }
    for i in disk_done:
        if not tracker["chunks"][i]["processed"]:
            mark_chunk_processed(tracker, i)
    if disk_done:
        save_tracker(out_dir, tracker)
        print(f"  Detectados {len(disk_done)} chunks existentes en disco.")

    # Chunks pendientes
    pending = {i for i in available if not tracker["chunks"][i]["processed"]} if resume else set(available)

    if not pending:
        print("Todos los chunks disponibles ya están procesados. Usa --merge.")
        save_tracker(out_dir, tracker)
        return

    print(f"\nIniciando extracción de {len(pending)} chunks...")

    config    = ModelConfig()
    extractor = OfflineExtractorV3(config=config)
    start_time = time.time()

    for done_count, i in enumerate(sorted(pending), 1):
        chunk = all_chunks[i]
        print(f"\n[{done_count}/{len(pending)}] {chunk['key']} ({chunk['num_trs']} TRs)")

        try:
            vit_feats, conf_feats, text_feats = process_chunk(
                extractor=extractor,
                mkv_path=chunk["mkv"],
                tsv_path=chunk["tsv"],
                num_trs_expected=chunk["num_trs"],
                tr_duration=tr_duration,
                text_only=text_only,
            )

            # Guardar los 3 tensores separados
            n  = vit_feats.shape[0]
            ok = (
                not vit_feats.isnan().any()
                and not conf_feats.isnan().any()
                and not text_feats.isnan().any()
            )
            torch.save(vit_feats,  chunks_dir / f"chunk_{i:03d}_vit.pt")
            torch.save(conf_feats, chunks_dir / f"chunk_{i:03d}_conformer.pt")
            torch.save(text_feats, chunks_dir / f"chunk_{i:03d}_text.pt")

            if ok:
                mark_chunk_processed(tracker, i, num_trs=n)
                print(f"  ✅ chunk_{i:03d} guardado ({n} TRs) | "
                      f"vit={tuple(vit_feats.shape)}, "
                      f"conformer={tuple(conf_feats.shape)}, "
                      f"text={tuple(text_feats.shape)}")
            else:
                mark_chunk_processed(tracker, i, error="NaN detectado")
                print(f"  ⚠️  chunk_{i:03d} con NaN. Guardado de todas formas.")

            save_tracker(out_dir, tracker)

            elapsed = time.time() - start_time
            eta_min = (len(pending) - done_count) * elapsed / done_count / 60
            print(f"  ETA: {eta_min:.0f} min ({done_count}/{len(pending)})")

        except Exception as e:
            print(f"\n  ERROR CRÍTICO en chunk {i}: {e}")
            mark_chunk_processed(tracker, i, error=str(e))
            save_tracker(out_dir, tracker)

    print("\nExtracción completa. Fusionando...")
    merge_chunks(output_dir, algonauts_dir, tracker=tracker)


def check_missing(algonauts_dir: str):
    all_chunks = get_chunks_info(algonauts_dir)
    missing = [(i, c["key"]) for i, c in enumerate(all_chunks)
               if not c["mkv"].exists() or not c["tsv"].exists()]
    if not missing:
        print(f"✅ Los {len(all_chunks)} chunks tienen estímulos.")
    else:
        print(f"⚠️  {len(missing)} chunks SIN estímulos:")
        for i, key in missing:
            print(f"  [{i:03d}] {key}")


def show_status(algonauts_dir, output_dir):
    out_dir = Path(output_dir)
    tracker_path = out_dir / "processed_chunks_v3.json"
    ci = get_chunks_info(algonauts_dir)
    if tracker_path.exists():
        with open(tracker_path) as f:
            tracker = json.load(f)
        print(f"Progreso: {tracker['processed_count']}/{tracker['total_chunks']} chunks")
        errors = [c for c in tracker["chunks"] if c.get("error")]
        if errors:
            print(f"  ⚠️  {len(errors)} chunks con errores")
    else:
        chunks_dir = out_dir / "chunks"
        done = sum(
            1 for i in range(len(ci))
            if all([
                (chunks_dir / f"chunk_{i:03d}_vit.pt").exists(),
                (chunks_dir / f"chunk_{i:03d}_conformer.pt").exists(),
                (chunks_dir / f"chunk_{i:03d}_text.pt").exists(),
            ])
        )
        print(f"Progreso: {done}/{len(ci)} chunks (sin tracker JSON)")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature Extractor V3: ViT + Conformer + Texto por TR"
    )
    parser.add_argument("--output_dir",     type=str, default="./data/features_v3")
    parser.add_argument("--algonauts_dir",  type=str, default="./algonauts_2025")
    parser.add_argument("--tr_duration",    type=float, default=1.49)
    parser.add_argument("--text_only",      action="store_true")
    parser.add_argument("--merge",          action="store_true")
    parser.add_argument("--status",         action="store_true")
    parser.add_argument("--check_missing",  action="store_true")
    parser.add_argument("--no_resume",      action="store_true")
    args = parser.parse_args()

    if args.check_missing:
        check_missing(args.algonauts_dir)
    elif args.merge:
        out_dir = Path(args.output_dir)
        tracker = None
        tp = out_dir / "processed_chunks_v3.json"
        if tp.exists():
            with open(tp) as f:
                tracker = json.load(f)
        merge_chunks(args.output_dir, args.algonauts_dir, tracker=tracker)
    elif args.status:
        show_status(args.algonauts_dir, args.output_dir)
    else:
        print("=" * 55)
        print("FEATURE EXTRACTOR V3 — ViT + Conformer + Texto")
        print(f"  Texto:   últimas {NARRATIVE_MAX_WORDS} palabras")
        print(f"  Audio:   {NARRATIVE_AUDIO_SEC:.0f}s previos")
        print(f"  Frames:  {NARRATIVE_NUM_FRAMES} equiespaciados (~32s)")
        print(f"  Grid:    {NARRATIVE_SUB_SAMPLES} Hz (1 muestra/TR)")
        print("=" * 55)
        generate_extraction(
            args.algonauts_dir,
            args.output_dir,
            args.tr_duration,
            text_only=args.text_only,
            resume=not args.no_resume,
        )
