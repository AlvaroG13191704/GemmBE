"""
prepare_train_test_split.py — Prepara datos con validación y test episódicos sin leakage.

Split riguroso por temporadas completas:
  TRAIN: Seasons 1-4 + Movies (~53k TRs, ~22h de contenido)
  VAL:   Season 5 de Friends (~22k TRs, ~9h de contenido, nunca visto en entrenamiento)
  TEST:  Season 6 de Friends (~23k TRs, ~9.5h de contenido, nunca visto en entrenamiento ni validación)

Aplica normalización z-score usando SOLO estadísticas del TRAIN set.
Esto elimina al 100% el data leakage de ventanas temporales (0% overlap)
y satisface el requerimiento del Reviewer #3.

Uso:
    uv run python -m src.prepare_train_test_split

Salida en data/train_test_split/:
    ├── features_train.pt
    ├── features_val.pt
    ├── features_test.pt
    ├── features_textonly_train.pt
    ├── features_textonly_val.pt
    ├── features_textonly_test.pt
    ├── bold_train_sub-01.pt
    ├── bold_val_sub-01.pt
    ├── bold_test_sub-01.pt
    ├── bold_train_sub-02.pt
    ├── bold_val_sub-02.pt
    ├── bold_test_sub-02.pt
    └── split_info.json
"""

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import torch


def identify_episodic_indices(tracker_path: Path) -> tuple[set[int], set[int], set[int]]:
    """
    Identifica los índices de chunks para Train (S1-S4 + Movies), Val (S5) y Test (S6).
    """
    with open(tracker_path, "r") as f:
        tracker = json.load(f)

    s6_test_indices = set()
    s5_val_indices = set()
    train_indices = set()

    for chunk in tracker["chunks"]:
        key = chunk["key"]
        idx = chunk["index"]
        if "task-s06" in key:
            s6_test_indices.add(idx)
        elif "task-s05" in key:
            s5_val_indices.add(idx)
        else:
            train_indices.add(idx)

    return train_indices, s5_val_indices, s6_test_indices


def _parse_chunk_key(key: str) -> tuple:
    """
    Parsea una clave de chunk y devuelve una tupla para ordenación cronológica.

    Formato esperado:
      - Friends: ses-XXX_task-s{season}e{episode}{part}
      - Movies:  ses-XXX_task-{name}{episode}_run-{run}

    Retorna tupla (category_order, season_or_name, episode, part_or_run)
    """
    friends_match = re.match(r"ses-\d+_task-s(\d+)e(\d+)([ab])", key)
    if friends_match:
        season = int(friends_match.group(1))
        episode = int(friends_match.group(2))
        part = friends_match.group(3)
        return (0, season, episode, part)

    movie_match = re.match(r"ses-\d+_task-([a-z]+)(\d+)(?:_run-(\d+))?", key)
    if movie_match:
        name = movie_match.group(1)
        episode = int(movie_match.group(2))
        run = int(movie_match.group(3)) if movie_match.group(3) else 1
        name_order = {"bourne": 0, "wolf": 1, "figures": 2, "life": 3}
        name_idx = name_order.get(name, 99)
        return (1, name_idx, episode, run)

    return (99, key, 0, "")


def load_and_split_features(
    chunks_dir: Path,
    train_indices: set[int],
    val_indices: set[int],
    test_indices: set[int],
    tracker_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Carga todos los chunks de features, los ordena cronológicamente,
    y los separa en train/val/test.
    """
    with open(tracker_path, "r") as f:
        tracker = json.load(f)

    chunk_info_by_index = {c["index"]: c for c in tracker["chunks"]}
    chunk_files = sorted(chunks_dir.glob("chunk_*.pt"))
    print(f"Chunks disponibles en {chunks_dir.parent.name}: {len(chunk_files)}")

    loaded_chunks = []
    for chunk_file in chunk_files:
        idx = int(chunk_file.stem.split("_")[1])
        tensor = torch.load(chunk_file, weights_only=True)
        key = chunk_info_by_index.get(idx, {}).get("key", f"unknown_{idx}")
        loaded_chunks.append((idx, key, tensor))

    loaded_chunks.sort(key=lambda x: _parse_chunk_key(x[1]))

    train_chunks = []
    val_chunks = []
    test_chunks = []
    chronological_order = []

    for idx, key, tensor in loaded_chunks:
        chronological_order.append({"index": idx, "key": key})
        if idx in test_indices:
            test_chunks.append(tensor)
        elif idx in val_indices:
            val_chunks.append(tensor)
        else:
            train_chunks.append(tensor)

    if not train_chunks:
        raise ValueError(f"No hay chunks de train en {chunks_dir}!")
    if not val_chunks:
        raise ValueError(f"No hay chunks de validación (Season 5) en {chunks_dir}!")
    if not test_chunks:
        raise ValueError(f"No hay chunks de test (Season 6) en {chunks_dir}!")

    features_train = torch.cat(train_chunks, dim=0)
    features_val = torch.cat(val_chunks, dim=0)
    features_test = torch.cat(test_chunks, dim=0)

    info = {
        "train_chunks": len(train_chunks),
        "val_chunks": len(val_chunks),
        "test_chunks": len(test_chunks),
        "chronological_order": chronological_order,
        "train_trs": features_train.shape[0],
        "val_trs": features_val.shape[0],
        "test_trs": features_test.shape[0],
    }

    return features_train, features_val, features_test, info


def load_and_split_fmri(
    fmri_path: Path,
    tracker_path: Path,
    train_indices: set[int],
    val_indices: set[int],
    test_indices: set[int],
    chronological_order: list[dict] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Carga fMRI de un sujeto y lo separa en train/val/test siguiendo los mismos chunks.
    """
    with open(tracker_path, "r") as f:
        tracker = json.load(f)

    chunk_info = {c["index"]: c for c in tracker["chunks"]}

    if chronological_order is not None:
        ordered_chunks = chronological_order
    else:
        ordered_chunks = [{"index": idx, "key": data["key"]} for idx, data in chunk_info.items()]

    with h5py.File(fmri_path, "r") as f:
        train_fmri = []
        val_fmri = []
        test_fmri = []

        for chunk_entry in ordered_chunks:
            idx = chunk_entry["index"]
            chunk_data = chunk_info.get(idx)

            if chunk_data is None or not chunk_data["processed"]:
                continue

            tracker_key = chunk_data["key"]
            task_suffix = tracker_key.split("_")[-1]

            matched_key = None
            for h5_key in f.keys():
                if h5_key.endswith(task_suffix):
                    matched_key = h5_key
                    break

            if matched_key is None:
                continue

            fmri_chunk = f[matched_key][:].astype(np.float32)
            if fmri_chunk.ndim == 1:
                fmri_chunk = fmri_chunk.reshape(1, -1)

            if fmri_chunk.shape[1] != 1000 and fmri_chunk.shape[0] == 1000:
                fmri_chunk = fmri_chunk.T

            num_trs_stimulus = chunk_data["num_trs_extracted"]
            if fmri_chunk.shape[0] > num_trs_stimulus:
                fmri_chunk = fmri_chunk[:num_trs_stimulus]

            if idx in test_indices:
                test_fmri.append(fmri_chunk)
            elif idx in val_indices:
                val_fmri.append(fmri_chunk)
            else:
                train_fmri.append(fmri_chunk)

    train_array = np.concatenate(train_fmri, axis=0) if train_fmri else np.empty((0, 1000), dtype=np.float32)
    val_array = np.concatenate(val_fmri, axis=0) if val_fmri else np.empty((0, 1000), dtype=np.float32)
    test_array = np.concatenate(test_fmri, axis=0) if test_fmri else np.empty((0, 1000), dtype=np.float32)

    info = {
        "train_trs": train_array.shape[0],
        "val_trs": val_array.shape[0],
        "test_trs": test_array.shape[0],
    }

    return train_array, val_array, test_array, info


def normalize_bold(
    train_bold: torch.Tensor,
    val_bold: torch.Tensor,
    test_bold: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Aplica z-score normalization usando SOLO estadísticas del train set.
    """
    mean = train_bold.mean(dim=0, keepdim=True)
    std = train_bold.std(dim=0, keepdim=True).clamp(min=1e-8)

    train_normalized = (train_bold - mean) / std
    val_normalized = (val_bold - mean) / std
    test_normalized = (test_bold - mean) / std

    stats = {
        "mean": mean.squeeze().tolist(),
        "std_mean": std.mean().item(),
        "std_min": std.min().item(),
        "std_max": std.max().item(),
    }

    return train_normalized, val_normalized, test_normalized, stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepara split train/val/test episódico (S1-S4+Movies -> Train, S5 -> Val, S6 -> Test)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/train_test_split",
        help="Directorio de salida",
    )
    parser.add_argument(
        "--fmri_dir",
        type=str,
        default="algonauts_2025/fmri",
        help="Directorio raíz con fMRI HDF5",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["sub-01", "sub-02"],
        help="Sujetos a procesar",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fmri_dir = Path(args.fmri_dir)

    print("=" * 60)
    print("PREPARANDO SPLIT TRAIN/VAL/TEST EPISÓDICO (Leakage-Free)")
    print("Train: Seasons 1-4 + Movies")
    print("Val:   Season 5 (Early Stopping limpio)")
    print("Test:  Season 6 (Hold-out final)")
    print("=" * 60)

    multimodal_tracker = Path("data/features/processed_chunks.json")
    textonly_tracker = Path("data/features_text_only/processed_chunks.json")

    train_idx, val_idx, test_idx = identify_episodic_indices(multimodal_tracker)
    print(f"\nChunks asignados:")
    print(f"  Train (S1-S4 + Movies): {len(train_idx)} chunks")
    print(f"  Val   (Season 5):       {len(val_idx)} chunks")
    print(f"  Test  (Season 6):       {len(test_idx)} chunks")

    # 1. Separar features Multimodales
    print(f"\n{'=' * 60}")
    print("Procesando FEATURES MULTIMODALES")
    print(f"{'=' * 60}")
    mm_train, mm_val, mm_test, mm_info = load_and_split_features(
        Path("data/features/chunks"), train_idx, val_idx, test_idx, multimodal_tracker
    )
    torch.save(mm_train, output_dir / "features_train.pt")
    torch.save(mm_val, output_dir / "features_val.pt")
    torch.save(mm_test, output_dir / "features_test.pt")
    print(f"  Train: {mm_info['train_trs']} TRs | Val: {mm_info['val_trs']} TRs | Test: {mm_info['test_trs']} TRs")

    # 2. Separar features Text-Only
    if textonly_tracker.exists() and Path("data/features_text_only/chunks").exists():
        print(f"\n{'=' * 60}")
        print("Procesando FEATURES TEXT-ONLY")
        print(f"{'=' * 60}")
        to_train, to_val, to_test, to_info = load_and_split_features(
            Path("data/features_text_only/chunks"), train_idx, val_idx, test_idx, textonly_tracker
        )
        torch.save(to_train, output_dir / "features_textonly_train.pt")
        torch.save(to_val, output_dir / "features_textonly_val.pt")
        torch.save(to_test, output_dir / "features_textonly_test.pt")
        print(f"  Train: {to_info['train_trs']} TRs | Val: {to_info['val_trs']} TRs | Test: {to_info['test_trs']} TRs")

    # 3. Procesar fMRI de cada sujeto
    all_info = {
        "train_indices": sorted(list(train_idx)),
        "val_indices": sorted(list(val_idx)),
        "test_indices": sorted(list(test_idx)),
        "multimodal_features": mm_info,
        "subjects": {},
        "normalization": "z-score per parcel (fit strictly on train, applied to train+val+test)",
    }

    for subject_id in args.subjects:
        print(f"\n{'=' * 60}")
        print(f"Procesando fMRI para {subject_id}")
        print(f"{'=' * 60}")

        func_dir = fmri_dir / subject_id / "func"
        friends_file = func_dir / f"{subject_id}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5"
        movie_file = func_dir / f"{subject_id}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5"

        train_fmri_parts = []
        val_fmri_parts = []
        test_fmri_parts = []

        for h5_path in [friends_file, movie_file]:
            if not h5_path.exists():
                print(f"  {h5_path.name} no encontrado, saltando")
                continue

            train_arr, val_arr, test_arr, fmri_info = load_and_split_fmri(
                h5_path, multimodal_tracker, train_idx, val_idx, test_idx,
                chronological_order=mm_info.get("chronological_order")
            )
            train_fmri_parts.append(train_arr)
            val_fmri_parts.append(val_arr)
            test_fmri_parts.append(test_arr)

        train_bold_np = np.concatenate(train_fmri_parts, axis=0)
        val_bold_np = np.concatenate(val_fmri_parts, axis=0)
        test_bold_np = np.concatenate(test_fmri_parts, axis=0)

        print(f"  fMRI train: {train_bold_np.shape[0]} TRs")
        print(f"  fMRI val:   {val_bold_np.shape[0]} TRs")
        print(f"  fMRI test:  {test_bold_np.shape[0]} TRs")

        train_bold = torch.from_numpy(train_bold_np).float()
        val_bold = torch.from_numpy(val_bold_np).float()
        test_bold = torch.from_numpy(test_bold_np).float()

        train_norm, val_norm, test_norm, norm_stats = normalize_bold(
            train_bold, val_bold, test_bold
        )

        torch.save(train_norm, output_dir / f"bold_train_{subject_id}.pt")
        torch.save(val_norm, output_dir / f"bold_val_{subject_id}.pt")
        torch.save(test_norm, output_dir / f"bold_test_{subject_id}.pt")

        all_info["subjects"][subject_id] = {
            "train_trs": train_norm.shape[0],
            "val_trs": val_norm.shape[0],
            "test_trs": test_norm.shape[0],
            "normalization_stats": norm_stats,
        }

    info_path = output_dir / "split_info.json"
    with open(info_path, "w") as f:
        json.dump(all_info, f, indent=2)

    print(f"\n{'=' * 60}")
    print("SPLIT COMPLETADO EXITOSAMENTE")
    print(f"Archivos guardados en: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
