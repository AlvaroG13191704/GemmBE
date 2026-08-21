"""
prepare_train_test_split.py — Prepara datos con Season 6 como hold-out estricto.

Separa:
  TRAIN: Seasons 1-5 + Movies (todo excepto Season 6 de Friends)
  TEST:  Season 6 de Friends (episodios completos, nunca vistos en entrenamiento)

Aplica normalización z-score usando SOLO estadísticas del TRAIN set.
Esto evita data leakage y es comparable con la metodología de TriBE v1.

Uso:
    uv run python -m src.prepare_train_test_split

Salida:
    data/train_test_split/
        ├── features_train.pt          # Features de entrenamiento
        ├── features_test.pt           # Features de test (S6)
        ├── bold_train_sub-01.pt       # fMRI train sub-01
        ├── bold_test_sub-01.pt        # fMRI test sub-01
        ├── bold_train_sub-02.pt
        ├── bold_test_sub-02.pt
        └── split_info.json            # Metadatos del split
"""

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import torch


def identify_season6_indices(tracker_path: Path) -> set[int]:
    """Identifica los índices de chunks que pertenecen a Season 6."""
    with open(tracker_path, "r") as f:
        tracker = json.load(f)

    s6_indices = set()
    for chunk in tracker["chunks"]:
        key = chunk["key"]
        # Season 6: task-s06eXX
        if "task-s06" in key:
            s6_indices.add(chunk["index"])

    return s6_indices


def _parse_chunk_key(key: str) -> tuple:
    """
    Parsea una clave de chunk y devuelve una tupla para ordenación cronológica.

    Formato esperado:
      - Friends: ses-XXX_task-s{season}e{episode}{part}
      - Movies:  ses-XXX_task-{name}{episode}_run-{run}

    Retorna tupla (category_order, season_or_name, episode, part_or_run)
    """
    # Intentar parsear Friends
    friends_match = re.match(r"ses-\d+_task-s(\d+)e(\d+)([ab])", key)
    if friends_match:
        season = int(friends_match.group(1))
        episode = int(friends_match.group(2))
        part = friends_match.group(3)
        return (0, season, episode, part)

    # Intentar parsear Movies
    movie_match = re.match(r"ses-\d+_task-([a-z]+)(\d+)(?:_run-(\d+))?", key)
    if movie_match:
        name = movie_match.group(1)
        episode = int(movie_match.group(2))
        run = int(movie_match.group(3)) if movie_match.group(3) else 1
        # Orden de películas: bourne, wolf, figures, life
        name_order = {"bourne": 0, "wolf": 1, "figures": 2, "life": 3}
        name_idx = name_order.get(name, 99)
        return (1, name_idx, episode, run)

    # Fallback: ordenar alfabéticamente al final
    return (99, key, 0, "")


def load_and_split_features(
    chunks_dir: Path,
    s6_indices: set[int],
    tracker_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Carga todos los chunks de features, los ordena cronológicamente,
    y los separa en train/test.

    Args:
        chunks_dir: Directorio con chunk_{i:03d}.pt
        s6_indices: Índices de chunks de Season 6
        tracker_path: Path al tracker JSON (necesario para orden cronológico)

    Returns:
        features_train, features_test, info
    """
    # Cargar tracker para obtener el orden cronológico
    with open(tracker_path, "r") as f:
        tracker = json.load(f)

    # Mapear índice -> key para ordenar
    chunk_info_by_index = {c["index"]: c for c in tracker["chunks"]}

    # Cargar todos los chunks disponibles
    chunk_files = sorted(chunks_dir.glob("chunk_*.pt"))
    print(f"Chunks disponibles: {len(chunk_files)}")

    loaded_chunks = []
    for chunk_file in chunk_files:
        idx = int(chunk_file.stem.split("_")[1])
        tensor = torch.load(chunk_file, weights_only=True)
        key = chunk_info_by_index.get(idx, {}).get("key", f"unknown_{idx}")
        loaded_chunks.append((idx, key, tensor))

    # Ordenar cronológicamente por key (no por índice de procesamiento)
    loaded_chunks.sort(key=lambda x: _parse_chunk_key(x[1]))

    train_chunks = []
    test_chunks = []
    train_indices = []
    test_indices = []
    chronological_order = []

    for idx, key, tensor in loaded_chunks:
        chronological_order.append({"index": idx, "key": key})
        if idx in s6_indices:
            test_chunks.append(tensor)
            test_indices.append(idx)
        else:
            train_chunks.append(tensor)
            train_indices.append(idx)

    if not train_chunks:
        raise ValueError("No hay chunks de entrenamiento!")
    if not test_chunks:
        raise ValueError("No hay chunks de test (Season 6)!")

    features_train = torch.cat(train_chunks, dim=0)
    features_test = torch.cat(test_chunks, dim=0)

    info = {
        "train_chunks": len(train_chunks),
        "test_chunks": len(test_chunks),
        "train_indices": train_indices,
        "test_indices": test_indices,
        "chronological_order": chronological_order,
        "train_trs": features_train.shape[0],
        "test_trs": features_test.shape[0],
    }

    return features_train, features_test, info


def load_and_split_fmri(
    fmri_path: Path,
    tracker_path: Path,
    s6_indices: set[int],
    chronological_order: list[dict] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Carga fMRI de un sujeto y lo separa en train/test siguiendo los mismos chunks.

    El fMRI se trunca EXACTAMENTE al número de TRs del chunk de estímulo
    para mantener sincronización perfecta.

    Args:
        chronological_order: Lista de dicts con 'index' y 'key' en orden cronológico.
            Si se proporciona, itera en ese orden. Si no, usa orden del tracker.
    """
    with open(tracker_path, "r") as f:
        tracker = json.load(f)

    # Build map: index -> chunk_data
    chunk_info = {c["index"]: c for c in tracker["chunks"]}

    # Determinar orden de iteración
    if chronological_order is not None:
        ordered_chunks = chronological_order
    else:
        ordered_chunks = [{"index": idx, "key": data["key"]} for idx, data in chunk_info.items()]

    with h5py.File(fmri_path, "r") as f:
        train_fmri = []
        test_fmri = []

        for chunk_entry in ordered_chunks:
            idx = chunk_entry["index"]
            chunk_data = chunk_info.get(idx)

            if chunk_data is None or not chunk_data["processed"]:
                continue

            # Match by task suffix (session numbers are subject-specific)
            tracker_key = chunk_data["key"]
            task_suffix = tracker_key.split("_")[-1]  # e.g., "task-s06e01a"

            # Find matching key in HDF5
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

            # Transpose if needed (should be (num_trs, 1000))
            if fmri_chunk.shape[1] != 1000 and fmri_chunk.shape[0] == 1000:
                fmri_chunk = fmri_chunk.T

            # Truncate to match stimulus TRs
            num_trs_stimulus = chunk_data["num_trs_extracted"]
            if fmri_chunk.shape[0] > num_trs_stimulus:
                fmri_chunk = fmri_chunk[:num_trs_stimulus]

            if idx in s6_indices:
                test_fmri.append(fmri_chunk)
            else:
                train_fmri.append(fmri_chunk)

    if not train_fmri:
        raise ValueError("No hay fMRI de entrenamiento!")

    train_array = np.concatenate(train_fmri, axis=0)
    if test_fmri:
        test_array = np.concatenate(test_fmri, axis=0)
    else:
        test_array = np.empty((0, train_array.shape[1]), dtype=np.float32)

    info = {
        "train_trs": train_array.shape[0],
        "test_trs": test_array.shape[0],
    }

    return train_array, test_array, info


def normalize_bold(
    train_bold: torch.Tensor,
    test_bold: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Aplica z-score normalization usando SOLO estadísticas del train set.
    Esto es CRÍTICO para evitar data leakage.
    """
    mean = train_bold.mean(dim=0, keepdim=True)
    std = train_bold.std(dim=0, keepdim=True).clamp(min=1e-8)

    train_normalized = (train_bold - mean) / std
    test_normalized = (test_bold - mean) / std  # Usa stats de train!

    stats = {
        "mean": mean.squeeze().tolist(),
        "std_mean": std.mean().item(),
        "std_min": std.min().item(),
        "std_max": std.max().item(),
    }

    return train_normalized, test_normalized, stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepara split train/test con Season 6 como hold-out"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/train_test_split",
        help="Directorio de salida",
    )
    parser.add_argument(
        "--chunks_dir",
        type=str,
        default="data/features/chunks",
        help="Directorio con chunk_{i:03d}.pt",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="data/features/processed_chunks.json",
        help="Tracker de chunks procesados",
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
    chunks_dir = Path(args.chunks_dir)
    tracker_path = Path(args.tracker)
    fmri_dir = Path(args.fmri_dir)

    print("=" * 60)
    print("PREPARANDO SPLIT TRAIN/TEST (Season 6 Hold-Out)")
    print("=" * 60)

    # 1. Identificar Season 6
    s6_indices = identify_season6_indices(tracker_path)
    print(f"\nSeason 6 chunks: {len(s6_indices)} chunks")
    print(f"Indices: {min(s6_indices)} - {max(s6_indices)}")

    # 2. Separar features
    print(f"\n{'=' * 60}")
    print("Procesando FEATURES")
    print(f"{'=' * 60}")

    features_train, features_test, feat_info = load_and_split_features(
        chunks_dir, s6_indices, tracker_path
    )

    print(f"  Train: {feat_info['train_chunks']} chunks, {feat_info['train_trs']} TRs")
    print(f"  Test:  {feat_info['test_chunks']} chunks, {feat_info['test_trs']} TRs")

    # 3. Procesar cada sujeto
    all_info = {
        "season6_indices": sorted(list(s6_indices)),
        "features": feat_info,
        "subjects": {},
        "normalization": "z-score per parcel (fit on train, apply to train+test)",
    }

    for subject_id in args.subjects:
        print(f"\n{'=' * 60}")
        print(f"Procesando {subject_id}")
        print(f"{'=' * 60}")

        # Load fMRI
        func_dir = fmri_dir / subject_id / "func"
        friends_file = func_dir / f"{subject_id}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5"
        movie_file = func_dir / f"{subject_id}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5"

        train_fmri_parts = []
        test_fmri_parts = []

        for h5_path in [friends_file, movie_file]:
            if not h5_path.exists():
                print(f"  {h5_path.name} no encontrado, saltando")
                continue

            train_arr, test_arr, fmri_info = load_and_split_fmri(
                h5_path, tracker_path, s6_indices,
                chronological_order=feat_info.get("chronological_order")
            )
            train_fmri_parts.append(train_arr)
            test_fmri_parts.append(test_arr)

        train_bold_np = np.concatenate(train_fmri_parts, axis=0)
        test_bold_np = np.concatenate(test_fmri_parts, axis=0) if test_fmri_parts else np.empty((0, train_bold_np.shape[1]), dtype=np.float32)

        print(f"  fMRI train: {train_bold_np.shape[0]} TRs")
        print(f"  fMRI test:  {test_bold_np.shape[0]} TRs")

        # Convert to tensors
        train_bold = torch.from_numpy(train_bold_np).float()
        test_bold = torch.from_numpy(test_bold_np).float()

        # Normalize (fit on train only!)
        train_bold_norm, test_bold_norm, norm_stats = normalize_bold(
            train_bold, test_bold
        )

        print(f"  Normalización: mean={np.mean(norm_stats['mean']):.4f}, "
              f"std_mean={norm_stats['std_mean']:.4f}")

        # Save
        torch.save(features_train, output_dir / "features_train.pt")
        torch.save(features_test, output_dir / "features_test.pt")
        torch.save(train_bold_norm, output_dir / f"bold_train_{subject_id}.pt")
        torch.save(test_bold_norm, output_dir / f"bold_test_{subject_id}.pt")

        all_info["subjects"][subject_id] = {
            "train_trs": train_bold_norm.shape[0],
            "test_trs": test_bold_norm.shape[0],
            "normalization_stats": norm_stats,
        }

    # Save metadata
    info_path = output_dir / "split_info.json"
    with open(info_path, "w") as f:
        json.dump(all_info, f, indent=2)

    print(f"\n{'=' * 60}")
    print("SPLIT COMPLETADO")
    print(f"{'=' * 60}")
    print(f"Directorio: {output_dir}")
    print(f"Metadatos: {info_path}")
    print(f"\nResumen:")
    print(f"  Train: {feat_info['train_trs']} TRs (~{feat_info['train_trs'] * 1.49 / 3600:.1f}h)")
    print(f"  Test:  {feat_info['test_trs']} TRs (~{feat_info['test_trs'] * 1.49 / 3600:.1f}h)")
    print(f"  Season 6 chunks: {len(s6_indices)}")
    print(f"  Sujetos: {', '.join(args.subjects)}")


if __name__ == "__main__":
    main()
