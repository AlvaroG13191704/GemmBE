"""
extract_features_v2.py — Codificador Narrativo con Gemma 4 para Algonauts 2025

Extracción de features multimodales (texto + audio + video) usando Gemma 4 E2B-it
congelado. Para cada TR, el modelo recibe:
  • TEXTO:     Últimas 1024 palabras del transcript
  • AUDIO:     30 segundos previos de audio a 16kHz
  • IMÁGENES:  32 frames equiespaciados de los últimos ~32 segundos

Cuadrícula: 1 Hz (1 muestra por TR de 1.49s).

Uso:
    # Extraer todos los chunks disponibles (default)
    python -m src.extract_features_v2

    # Solo texto (baseline)
    python -m src.extract_features_v2 --text_only

    # Verificar qué chunks faltan por descargar
    python -m src.extract_features_v2 --check_missing

    # Fusionar chunks ya procesados
    python -m src.extract_features_v2 --merge

    # Ver progreso
    python -m src.extract_features_v2 --status
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
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import h5py
from transformers import AutoProcessor, AutoModelForMultimodalLM

# pyrefly: ignore [missing-import]
from src.config import ModelConfig
# pyrefly: ignore [missing-import]
from src.utils.temporal_alignment import TemporalPooling
# pyrefly: ignore [missing-import]
from src.utils.prepare_fmri import get_fmri_paths, load_algonauts_fmri
# pyrefly: ignore [missing-import]
from src.utils.helpers import extract_audio_from_mkv


# =============================================================================
# Constantes del Codificador Narrativo
# =============================================================================
NARRATIVE_MAX_WORDS = 1024
NARRATIVE_AUDIO_SEC = 30.0
NARRATIVE_NUM_FRAMES = 32
NARRATIVE_SUB_SAMPLES = 1  # 1 Hz


# =============================================================================
# OfflineExtractor — Motor de extracción con Gemma 4 congelado
# =============================================================================

class OfflineExtractor:
    def __init__(self, config: ModelConfig = None, multi_layer: bool = True):
        self.config = config or ModelConfig()
        self.multi_layer = multi_layer
        device = self.config.device
        dtype = self.config.dtype

        print("=" * 50)
        print("Offline Feature Extractor (Algonauts)")
        print(f"  Device: {device}")
        print(f"  Dtype:  {dtype}")
        mode_str = "Multi-Layer" if multi_layer else str(self.config.extraction_layer)
        print(f"  Mode:   {mode_str}")

        self._load_gemma(device, dtype)

        self.num_layers = 35
        self.global_layer_indices = [i + 1 for i in range(4, self.num_layers, 5)]
        if self.multi_layer:
            self.block1_indices = [i for i in self.global_layer_indices if 20 <= i <= 25]
            self.block2_indices = [i for i in self.global_layer_indices if 30 <= i <= 35]

        self.temporal_pooling = TemporalPooling(
            hidden_size=self.config.gemma_hidden_size, dtype=dtype,
        ).to(device)

        self._hooks = []
        self._hook_outputs = {}
        self._register_hooks()

    def _load_gemma(self, device: torch.device, dtype: torch.dtype) -> None:
        print(f"  Loading {self.config.model_id}...")
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
        print(f"  Gemma 4 loaded ({total_params:,} parameters)")

    def _get_transformer_layers(self):
        m = self.model
        if hasattr(m, 'model'):
            m = m.model
        if hasattr(m, 'language_model'):
            m = m.language_model
        if hasattr(m, 'model'):
            m = m.model
        return m.layers

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self._hook_outputs[layer_idx] = hidden
        return hook

    def _register_hooks(self):
        if self.multi_layer:
            target_layers = set(self.block1_indices + self.block2_indices)
        else:
            extraction = self.config.extraction_layer
            if extraction < 0:
                extraction = self.num_layers
            target_layers = {extraction}

        layers = self._get_transformer_layers()
        for layer_idx in target_layers:
            module = layers[layer_idx - 1]
            h = module.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(h)

    def _pool_multi_layer(self, hidden_states, attention_mask=None):
        b1 = torch.stack([hidden_states[i] for i in self.block1_indices]).mean(0)
        b2 = torch.stack([hidden_states[i] for i in self.block2_indices]).mean(0)
        combined = (b1 + b2) / 2.0
        m = combined.mean(dim=-1, keepdim=True)
        s = combined.std(dim=-1, keepdim=True) + 1e-6
        return self.temporal_pooling((combined - m) / s, attention_mask)

    def _pool_single_layer(self, hidden_states, attention_mask=None):
        return self.temporal_pooling(hidden_states[self.config.extraction_layer], attention_mask)

    def _run_forward_and_pool(self, inputs: dict):
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.config.dtype)
        if "audio_values" in inputs:
            inputs["audio_values"] = inputs["audio_values"].to(self.config.dtype)

        self._hook_outputs.clear()
        outputs = self.model(**inputs, output_hidden_states=False)

        hs_list = [None] * (self.num_layers + 1)
        for idx, tensor in self._hook_outputs.items():
            hs_list[idx] = tensor

        if self.multi_layer:
            pooled = self._pool_multi_layer(hs_list, inputs.get("attention_mask"))
        else:
            pooled = self._pool_single_layer(hs_list, inputs.get("attention_mask"))

        del inputs, outputs, hs_list
        self._hook_outputs.clear()
        return pooled

    def _extract_single_impl(self, text: str, images: list = None, audio: np.ndarray = None) -> torch.Tensor:
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

    @torch.inference_mode()
    def extract_single(self, text: str, images: list = None, audio: np.ndarray = None) -> torch.Tensor:
        try:
            return self._extract_single_impl(text, images, audio)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            oom_str = str(e).lower()
            is_oom = "out of memory" in oom_str or "cuda" in oom_str
            if not is_oom:
                raise
            if images and len(images) > 16:
                print(f"    OOM with {len(images)} imgs, retrying with 16...")
                torch.cuda.empty_cache()
                return self._extract_single_impl(text, images[-16:], audio)
            else:
                print(f"    OOM even with <=16 imgs, returning zeros.")
                torch.cuda.empty_cache()
                return torch.zeros(self.config.gemma_hidden_size)


# =============================================================================
# Pre-extracción de frames
# =============================================================================

def _preextract_frames_sequential(mkv_path: Path, needed_indices: set, fps: float, total_frames: int):
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
# process_chunk — Extracción completa de un chunk
# =============================================================================

def process_chunk(
    extractor: OfflineExtractor,
    mkv_path: Path,
    tsv_path: Path,
    num_trs_expected: int,
    tr_duration: float,
    text_only: bool = False,
) -> torch.Tensor:
    sr = 16000

    # 1. Cargar texto por TR
    df = pd.read_csv(tsv_path, sep='\t')
    if len(df) != num_trs_expected:
        print(f"  WARNING: TSV TRs ({len(df)}) != HDF5 TRs ({num_trs_expected})")
        num_trs_expected = min(len(df), num_trs_expected)

    all_words = []
    cumulative = []
    for i in range(num_trs_expected):
        t = str(df.iloc[i]['text_per_tr'])
        if t != "nan" and t.strip() not in ("", "[]"):
            cumulative.extend(t.strip().split())
        all_words.append(list(cumulative))

    # 2. Cargar audio
    print(f"  Extracting audio from {mkv_path.name}...")
    audio_waveform = extract_audio_from_mkv(mkv_path, sr=sr)
    if len(audio_waveform) <= 1:
        print("  WARNING: Empty audio, using silence.")
        audio_waveform = np.zeros(int(num_trs_expected * tr_duration * sr))

    # 3. Video metadata
    cap = cv2.VideoCapture(str(mkv_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    VISUAL_WINDOW_SEC = 32.0

    # 4. Precalcular frames necesarios
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
        print("  TEXT-ONLY mode: skipping video and audio.")
    else:
        print(f"  Pre-extracting {len(needed_frame_indices)} unique frames...")
        frame_cache = _preextract_frames_sequential(mkv_path, needed_frame_indices, fps, total_frames)

    def _get_frame(fidx: int):
        raw = frame_cache.get(fidx)
        if raw is None:
            return None
        return Image.open(io.BytesIO(raw))

    # 6. Loop principal (1 Hz = 1 muestra por TR)
    error_count = 0
    chunk_features = []

    print(f"  Processing {mkv_path.name} ({num_trs_expected} TRs)...")

    for tr_idx in tqdm(range(num_trs_expected), desc="TRs", leave=False):
        current_time_sec = (tr_idx + 1) * tr_duration
        words = all_words[tr_idx]
        text_prompt = " ".join(words[-NARRATIVE_MAX_WORDS:]) if words else ""

        if text_only:
            images, audio_segment = None, None
        else:
            # Audio
            a_end = int(current_time_sec * sr)
            a_start = int(max(0, current_time_sec - NARRATIVE_AUDIO_SEC) * sr)
            a_end = min(a_end, len(audio_waveform))
            audio_segment = audio_waveform[a_start:a_end] if a_start < a_end else np.zeros(int(tr_duration * sr))

            # Images
            window_start = max(0.0, current_time_sec - VISUAL_WINDOW_SEC)
            images_for_gemma = []
            for fi in range(NARRATIVE_NUM_FRAMES):
                t_sec = window_start + (fi + 0.5) * (current_time_sec - window_start) / NARRATIVE_NUM_FRAMES
                fidx = min(int(t_sec * fps), total_frames - 1)
                fidx = max(0, fidx)
                img = _get_frame(fidx)
                if img is not None:
                    images_for_gemma.append(img)
            images = images_for_gemma if images_for_gemma else None

        try:
            pooled = extractor.extract_single(text=text_prompt, images=images, audio=audio_segment)
            if pooled.isnan().any():
                pooled = torch.zeros(1536)
                error_count += 1
            chunk_features.append(pooled)
        except Exception as e:
            if error_count < 5:
                print(f"\n  ERROR in TR {tr_idx}: {e}")
            chunk_features.append(torch.zeros(1536))
            error_count += 1

    if error_count > 0:
        print(f"  WARNING: {error_count} extraction errors in this chunk.")
    return torch.stack(chunk_features, dim=0)


# =============================================================================
# get_chunks_info — Lista todos los chunks del dataset
# =============================================================================

def get_chunks_info(algonauts_dir: str) -> list[dict]:
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


# =============================================================================
# Tracker JSON
# =============================================================================

def load_or_create_chunk_tracker(output_dir: Path, chunks_info: list[dict], tracker_path: Path = None) -> dict:
    if tracker_path is None:
        tracker_path = output_dir / "processed_chunks.json"
    if tracker_path.exists():
        with open(tracker_path, 'r') as f:
            tracker = json.load(f)
        if len(tracker.get("chunks", [])) == len(chunks_info):
            return tracker
        print("  WARNING: Tracker JSON outdated (chunk count changed). Restarting...")

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
    tracker_path = output_dir / "processed_chunks.json"
    with open(tracker_path, 'w') as f:
        json.dump(tracker, f, indent=2, ensure_ascii=False)


def mark_chunk_processed(tracker: dict, chunk_index: int, error: str = None, num_trs_extracted: int = None) -> None:
    entry = tracker["chunks"][chunk_index]
    entry["processed"] = error is None
    entry["timestamp"] = datetime.now().isoformat() if error is None else entry["timestamp"]
    entry["error"] = error
    if num_trs_extracted is not None:
        entry["num_trs_extracted"] = num_trs_extracted
    tracker["processed_count"] = sum(1 for c in tracker["chunks"] if c["processed"])
    tracker["last_updated"] = datetime.now().isoformat()


# =============================================================================
# merge_chunks
# =============================================================================

def merge_chunks(output_dir: str, algonauts_dir: str, tracker: dict = None):
    out_dir = Path(output_dir)
    chunks_dir = out_dir / "chunks"
    chunks_info = get_chunks_info(algonauts_dir)

    if tracker is not None:
        indices_to_merge = [i for i, c in enumerate(tracker["chunks"]) if c["processed"]]
        expected_trs = sum(
            tracker["chunks"][i].get("num_trs_extracted") or tracker["chunks"][i]["num_trs"]
            for i in indices_to_merge
        )
        print(f"\nMerging {len(indices_to_merge)}/{len(chunks_info)} processed chunks...")
    else:
        indices_to_merge = list(range(len(chunks_info)))
        expected_trs = sum(c["num_trs"] for c in chunks_info)
        print(f"\nMerging {len(indices_to_merge)} chunks...")

    all_tensors, missing = [], []
    for i in indices_to_merge:
        p = chunks_dir / f"chunk_{i:03d}.pt"
        if not p.exists():
            missing.append(i)
            continue
        all_tensors.append(torch.load(p, weights_only=True))

    if missing:
        print(f"  WARNING: Missing {len(missing)} .pt files: {missing[:20]}")
        return

    if not all_tensors:
        print("  WARNING: No tensors to merge.")
        return

    final = torch.cat(all_tensors, dim=0)
    out_path = out_dir / "real_stimulus_features.pt"
    torch.save(final, out_path)
    print(f"  Saved: {out_path} -> {final.shape}")
    if final.shape[0] == expected_trs:
        print("  Perfect sync!")
    else:
        print(f"  WARNING: Difference of {abs(final.shape[0] - expected_trs)} TRs")
    return final


# =============================================================================
# generate_real_extraction — Orquestador principal
# =============================================================================

def generate_real_extraction(
    algonauts_dir, output_dir, tr_duration,
    multi_layer,
    text_only=False,
    resume=True,
):
    out_dir = Path(output_dir)
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    print("\nScanning dataset...")
    all_chunks = get_chunks_info(algonauts_dir)
    total_trs = sum(c["num_trs"] for c in all_chunks)
    print(f"  {len(all_chunks)} total chunks, {total_trs:,} TRs.")

    # Cargar tracker existente
    tracker_path = out_dir / "processed_chunks.json"
    tracker = load_or_create_chunk_tracker(out_dir, all_chunks, tracker_path)
    processed_before = tracker["processed_count"]

    # Verificar qué chunks tienen estímulos disponibles
    available_indices = set()
    missing_stimuli = []
    for i, chunk in enumerate(all_chunks):
        has_mkv = chunk["mkv"].exists()
        has_tsv = chunk["tsv"].exists()
        if has_mkv and has_tsv:
            available_indices.add(i)
        else:
            missing_stimuli.append(i)

    if missing_stimuli:
        print(f"\n  WARNING: {len(missing_stimuli)} chunks MISSING stimuli (skipped):")
        for idx in missing_stimuli[:10]:
            c = all_chunks[idx]
            missing = []
            if not c["mkv"].exists():
                missing.append("video")
            if not c["tsv"].exists():
                missing.append("transcript")
            print(f"    [{idx:03d}] {c['key']} -> missing: {', '.join(missing)}")
        if len(missing_stimuli) > 10:
            print(f"    ... and {len(missing_stimuli) - 10} more")
        print(f"\n  To download missing stimuli, use:")
        print(f"    cd algonauts_2025 && datalad get <paths_above>")

    # Detectar chunks ya en disco
    disk_done = {i for i in available_indices if (chunks_dir / f"chunk_{i:03d}.pt").exists()}
    for i in disk_done:
        if not tracker["chunks"][i]["processed"]:
            mark_chunk_processed(tracker, i)
    if tracker["processed_count"] > processed_before:
        save_chunk_tracker(out_dir, tracker)
        print(f"  Detected {tracker['processed_count'] - processed_before} existing chunks on disk.")

    # Determinar chunks pendientes
    if resume:
        pending_indices = {i for i in available_indices if not tracker["chunks"][i]["processed"]}
    else:
        pending_indices = set(available_indices)

    pending = len(pending_indices)
    if tracker["processed_count"] > 0:
        print(f"  RESUME: {tracker['processed_count']} done, {pending} pending.")
    if pending == 0:
        print("All available chunks processed. Use --merge.")
        save_chunk_tracker(out_dir, tracker)
        return

    print(f"\nStarting extraction of {pending} chunks...")

    config = ModelConfig()
    extractor = OfflineExtractor(config=config, multi_layer=multi_layer)

    start_time = time.time()
    done_count = 0
    for i in sorted(pending_indices):
        chunk = all_chunks[i]
        print(f"\n[{i+1}/{len(all_chunks)}] {chunk['key']} ({chunk['num_trs']} TRs)")
        try:
            features = process_chunk(
                extractor=extractor, mkv_path=chunk["mkv"], tsv_path=chunk["tsv"],
                num_trs_expected=chunk["num_trs"], tr_duration=tr_duration,
                text_only=text_only,
            )

            # Guardar
            chunk_path = chunks_dir / f"chunk_{i:03d}.pt"
            num_trs = features.shape[0]
            has_nan = torch.isnan(features).any().item()
            has_inf = torch.isinf(features).any().item()
            zero_trs = (features.abs().sum(dim=-1) == 0).sum().item()
            valid = not has_nan and not has_inf and zero_trs < num_trs

            if not valid:
                print(f"  WARNING: Corrupt chunk: NaN={has_nan} Inf={has_inf} zeros={zero_trs}/{num_trs}")
                torch.save(features, chunk_path)
                mark_chunk_processed(tracker, i, error=f"corrupt: nan={has_nan} inf={has_inf} zeros={zero_trs}")
            else:
                torch.save(features, chunk_path)
                mark_chunk_processed(tracker, i, num_trs_extracted=num_trs)
                if zero_trs > 0:
                    print(f"  WARNING: {zero_trs}/{num_trs} TRs with empty embeddings")
                print(f"  Saved chunk_{i:03d}.pt ({num_trs} TRs)")

            save_chunk_tracker(out_dir, tracker)
            done_count += 1
            elapsed = time.time() - start_time
            eta = (pending - done_count) * elapsed / done_count / 60 if done_count > 0 else 0
            print(f"  ETA remaining: {eta:.0f} min ({done_count}/{pending} chunks)")
        except Exception as e:
            print(f"\n  CRITICAL ERROR in chunk {i}: {e}")
            mark_chunk_processed(tracker, i, error=str(e))
            save_chunk_tracker(out_dir, tracker)

    print("\nAll available chunks processed. Merging...")
    merge_chunks(output_dir, algonauts_dir, tracker=tracker)


# =============================================================================
# check_missing — Muestra qué chunks faltan por descargar
# =============================================================================

def check_missing(algonauts_dir: str):
    print("\nChecking for missing stimuli...")
    all_chunks = get_chunks_info(algonauts_dir)
    missing = []
    for i, chunk in enumerate(all_chunks):
        has_mkv = chunk["mkv"].exists()
        has_tsv = chunk["tsv"].exists()
        if not has_mkv or not has_tsv:
            missing.append((i, chunk["key"], has_mkv, has_tsv))

    if not missing:
        print(f"  All {len(all_chunks)} chunks have stimuli available!")
        return

    print(f"\n  {len(missing)} chunks MISSING stimuli:")
    for idx, key, has_mkv, has_tsv in missing:
        missing_parts = []
        if not has_mkv:
            missing_parts.append("video (.mkv)")
        if not has_tsv:
            missing_parts.append("transcript (.tsv)")
        print(f"    [{idx:03d}] {key} -> missing: {', '.join(missing_parts)}")

    print(f"\n  Download commands (run from algonauts_2025/):")
    for idx, key, has_mkv, has_tsv in missing:
        chunk = all_chunks[idx]
        if not has_mkv:
            rel = str(chunk["mkv"].relative_to(Path(algonauts_dir)))
            print(f"    datalad get {rel}")
        if not has_tsv:
            rel = str(chunk["tsv"].relative_to(Path(algonauts_dir)))
            print(f"    datalad get {rel}")


# =============================================================================
# status — Muestra progreso actual
# =============================================================================

def show_status(algonauts_dir: str, output_dir: str):
    ci = get_chunks_info(algonauts_dir)
    out_dir = Path(output_dir)
    tracker_path = out_dir / "processed_chunks.json"

    if tracker_path.exists():
        with open(tracker_path, 'r') as f:
            tracker = json.load(f)
        total_trs = sum(
            c.get("num_trs_extracted") or c["num_trs"]
            for c in tracker["chunks"] if c["processed"]
        )
        print(f"\nProgress: {tracker['processed_count']}/{tracker['total_chunks']} chunks processed")
        print(f"  TRs covered: {total_trs:,}")
        errors = [c for c in tracker["chunks"] if c.get("error")]
        if errors:
            print(f"  WARNING: {len(errors)} chunks with errors")
    else:
        cd = out_dir / "chunks"
        done = sum(1 for i in range(len(ci)) if (cd / f"chunk_{i:03d}.pt").exists())
        print(f"\nProgress: {done}/{len(ci)} chunks processed (no tracker JSON)")

    # También mostrar cuántos faltan por descargar
    missing = sum(1 for c in ci if not c["mkv"].exists() or not c["tsv"].exists())
    if missing:
        print(f"  WARNING: {missing} chunks missing stimuli")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Narrative Encoder v2: Gemma 4 x Algonauts 2025"
    )
    parser.add_argument("--output_dir", type=str, default="./data/features_v2")
    parser.add_argument("--algonauts_dir", type=str, default="./algonauts_2025")
    parser.add_argument("--tr_duration", type=float, default=1.49)
    parser.add_argument("--single_layer", action="store_true",
                        help="Extract only last layer instead of multi-layer")
    parser.add_argument("--text_only", action="store_true",
                        help="Text-only mode: skip images and audio")
    parser.add_argument("--merge", action="store_true",
                        help="Only merge already processed chunks")
    parser.add_argument("--status", action="store_true",
                        help="Show extraction progress from tracker JSON")
    parser.add_argument("--check_missing", action="store_true",
                        help="Show which chunks are missing stimuli (no extraction)")
    parser.add_argument("--no_resume", action="store_true",
                        help="Re-process all available chunks (ignore tracker)")

    args = parser.parse_args()

    if args.check_missing:
        check_missing(args.algonauts_dir)

    elif args.merge:
        out_dir = Path(args.output_dir)
        tracker = None
        tracker_path = out_dir / "processed_chunks.json"
        if tracker_path.exists():
            with open(tracker_path, 'r') as f:
                tracker = json.load(f)
            print(f"Tracker loaded: {tracker['processed_count']}/{tracker['total_chunks']} chunks.")
        merge_chunks(args.output_dir, args.algonauts_dir, tracker=tracker)

    elif args.status:
        show_status(args.algonauts_dir, args.output_dir)

    else:
        print("=" * 50)
        print("NARRATIVE ENCODER v2")
        print(f"  Text:    last {NARRATIVE_MAX_WORDS} words")
        print(f"  Audio:   {NARRATIVE_AUDIO_SEC:.0f}s previous")
        print(f"  Frames:  {NARRATIVE_NUM_FRAMES} evenly spaced (~32s)")
        print(f"  Grid:    {NARRATIVE_SUB_SAMPLES} Hz (1 sample per TR)")
        print("=" * 50)
        generate_real_extraction(
            args.algonauts_dir, args.output_dir, args.tr_duration,
            not args.single_layer,
            text_only=args.text_only,
            resume=not args.no_resume,
        )
