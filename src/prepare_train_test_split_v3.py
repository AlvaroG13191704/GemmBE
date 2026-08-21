"""
prepare_train_test_split_v3.py — Prepara splits train/test para v3 (3 modalidades separadas).

Separa:
  TRAIN: Seasons 1-5 + Movies (todo excepto Season 6 de Friends)
  TEST:  Season 6 de Friends (episodios completos, hold-out estricto)

Normaliza BOLD usando estadísticas de TRAIN únicamente para evitar leakage.
Normaliza features (vit, conformer, text) por dimensión usando estadísticas de TRAIN.

Uso:
    python -m src.prepare_train_test_split_v3
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


def load_and_split_modality(
    chunks_dir: Path,
    s6_indices: set[int],
    tracker_path: Path,
    modality: str,
) -> tuple[torch.Tensor, torch.Tensor, list]:
    """
    Carga todos los chunks de features para una modalidad específica,
    los ordena cronológicamente y los separa en train/test.
    """
    with open(tracker_path, "r") as f:
        tracker = json.load(f)

    chunk_info_by_index = {c["index"]: c for c in tracker["chunks"]}

    # Cargar chunks disponibles para esta modalidad
    chunk_files = sorted(chunks_dir.glob(f"chunk_*_{modality}.pt"))
    if not chunk_files:
        raise FileNotFoundError(f"No se encontraron archivos chunk_*_{modality}.pt en {chunks_dir}")

    loaded_chunks = []
    for chunk_file in chunk_files:
        # Extraer el índice del nombre del archivo (ej: chunk_003_vit.pt -> 3)
        parts = chunk_file.stem.split("_")
        idx = int(parts[1])
        tensor = torch.load(chunk_file, weights_only=True)
        key = chunk_info_by_index.get(idx, {}).get("key", f"unknown_{idx}")
        loaded_chunks.append((idx, key, tensor))

    # Ordenar cronológicamente
    loaded_chunks.sort(key=lambda x: _parse_chunk_key(x[1]))

    train_chunks = []
    test_chunks = []
    chronological_order = []

    for idx, key, tensor in loaded_chunks:
        chronological_order.append({"index": idx, "key": key})
        if idx in s6_indices:
            test_chunks.append(tensor)
        else:
            train_chunks.append(tensor)

    if not train_chunks:
        raise ValueError(f"No hay chunks de entrenamiento para {modality}!")
    if not test_chunks:
        raise ValueError(f"No hay chunks de test (S6) para {modality}!")

    features_train = torch.cat(train_chunks, dim=0)
    features_test = torch.cat(test_chunks, dim=0)

    return features_train, features_test, chronological_order


def load_and_split_fmri(
    fmri_path: Path,
    tracker_path: Path,
    s6_indices: set[int],
    chronological_order: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Carga fMRI de un sujeto y lo separa en train/test cronológicamente.
    """
    with open(tracker_path, "r") as f:
        tracker = json.load(f)

    chunk_info = {c["index"]: c for c in tracker["chunks"]}

    with h5py.File(fmri_path, "r") as f:
        train_fmri = []
        test_fmri = []

        for chunk_entry in chronological_order:
            idx = chunk_entry["index"]
            chunk_data = chunk_info.get(idx)

            if chunk_data is None or not chunk_data.get("processed", False):
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

            # Truncar al número de TRs extraídos
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
    test_array = np.concatenate(test_fmri, axis=0) if test_fmri else np.empty((0, train_array.shape[1]), dtype=np.float32)

    return train_array, test_array


def main():
    parser = argparse.ArgumentParser(
        description="Prepara split train/test para v3 (multimodal)"
    )
    parser.add_argument("--output_dir", type=str, default="data/train_test_split_v3")
    parser.add_argument("--chunks_dir", type=str, default="data/features_v3/chunks")
    parser.add_argument("--tracker", type=str, default="data/features_v3/processed_chunks_v3.json")
    parser.add_argument("--fmri_dir", type=str, default="algonauts_2025/fmri")
    parser.add_argument("--subjects", nargs="+", default=["sub-01", "sub-02"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = Path(args.chunks_dir)
    tracker_path = Path(args.tracker)
    fmri_dir = Path(args.fmri_dir)

    print("=" * 60)
    print("PREPARANDO SPLIT V3 TRAIN/TEST (3 Modalidades Separadas)")
    print("=" * 60)

    if not tracker_path.exists():
        # Intentar fallback si estamos testeando localmente sin tracker v3
        fallback = tracker_path.parent.parent / "features/processed_chunks.json"
        if fallback.exists():
            print(f"  Tracker v3 no encontrado. Usando fallback: {fallback}")
            tracker_path = fallback
        else:
            raise FileNotFoundError(f"No se encontró el tracker en {tracker_path}")

    # 1. Identificar Season 6
    s6_indices = identify_season6_indices(tracker_path)
    print(f"\nSeason 6 chunks: {len(s6_indices)} chunks")

    # 2. Separar cada modalidad
    modalities = ["vit", "conformer", "text"]
    split_data = {}
    chronological_order = None

    for mod in modalities:
        print(f"\nProcesando modalidad: {mod.upper()}")
        try:
            train_feats, test_feats, ch_order = load_and_split_modality(
                chunks_dir, s6_indices, tracker_path, mod
            )
            split_data[f"{mod}_train"] = train_feats
            split_data[f"{mod}_test"] = test_feats
            
            # Guardar el orden cronológico del primer feature
            if chronological_order is None:
                chronological_order = ch_order
                
            print(f"  Train: {train_feats.shape} | Test: {test_feats.shape}")
            
            # Guardar en disco
            torch.save(train_feats, output_dir / f"{mod}_train.pt")
            torch.save(test_feats, output_dir / f"{mod}_test.pt")
        except Exception as e:
            print(f"  ⚠️ Error al procesar {mod}: {e}")

    # 3. Procesar fMRI de cada sujeto
    all_info = {
        "season6_indices": sorted(list(s6_indices)),
        "chronological_order": chronological_order,
        "subjects": {},
        "normalization": "z-score per parcel (fit on train, apply to train+test)",
    }

    for subject_id in args.subjects:
        print(f"\nProcesando sujeto: {subject_id}")
        func_dir = fmri_dir / subject_id / "func"
        friends_file = func_dir / f"{subject_id}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5"
        movie_file = func_dir / f"{subject_id}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5"

        train_parts, test_parts = [], []

        for h5_path in [friends_file, movie_file]:
            if not h5_path.exists():
                print(f"  {h5_path.name} no encontrado, saltando")
                continue

            try:
                train_arr, test_arr = load_and_split_fmri(
                    h5_path, tracker_path, s6_indices, chronological_order
                )
                train_parts.append(train_arr)
                test_parts.append(test_arr)
            except Exception as e:
                print(f"  ⚠️ Error en H5 {h5_path.name}: {e}")

        if not train_parts:
            print(f"  ❌ No se pudo cargar fMRI para {subject_id}")
            continue

        train_bold_np = np.concatenate(train_parts, axis=0)
        test_bold_np = np.concatenate(test_parts, axis=0) if test_parts else np.empty((0, train_bold_np.shape[1]), dtype=np.float32)

        print(f"  fMRI train: {train_bold_np.shape} | Test: {test_bold_np.shape}")

        train_bold = torch.from_numpy(train_bold_np).float()
        test_bold = torch.from_numpy(test_bold_np).float()

        # Normalizar BOLD usando estadísticas del train set
        mean = train_bold.mean(dim=0, keepdim=True)
        std = train_bold.std(dim=0, keepdim=True).clamp(min=1e-8)

        train_bold_norm = (train_bold - mean) / std
        test_bold_norm = (test_bold - mean) / std

        # Guardar en disco
        torch.save(train_bold_norm, output_dir / f"bold_train_{subject_id}.pt")
        torch.save(test_bold_norm, output_dir / f"bold_test_{subject_id}.pt")

        all_info["subjects"][subject_id] = {
            "train_trs": train_bold_norm.shape[0],
            "test_trs": test_bold_norm.shape[0],
        }

    # Guardar metadatos
    info_path = output_dir / "split_info.json"
    with open(info_path, "w") as f:
        json.dump(all_info, f, indent=2)

    print(f"\n✅ SPLIT V3 COMPLETADO")
    print(f"Directorio de salida: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
