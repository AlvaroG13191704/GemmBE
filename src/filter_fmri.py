"""
Filtra datos fMRI de Algonauts 2025 para incluir SOLO los chunks
que fueron procesados por el extractor de estímulos.

Uso:
    # Filtrar sub-01 y sub-02 (default)
    python src/filter_fmri.py

    # Filtrar sujetos específicos
    python src/filter_fmri.py --subjects sub-01 sub-02 sub-03

    # Filtrar solo un sujeto
    python src/filter_fmri.py --subject_id sub-01

    # Especificar tracker y output manualmente
    python src/filter_fmri.py \
        --tracker ./data/features/processed_chunks.json \
        --output_dir ./data/subjects_fmri_filtered

El script genera tensores .pt con la misma forma que prepare_fmri.py,
pero solo incluyendo los TRs de los chunks que existen en el tracker.
Esto permite entrenar el modelo SOLO con el subset de datos que
realmente tienen features de estímulo extraídos.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
# pyrefly: ignore [missing-import]
from src.utils.prepare_fmri import (
    NUM_PARCELS,
    get_fmri_paths,
)
# pyrefly: ignore [missing-import]
from src.extract_features_v2 import get_chunks_info


def _task_from_key(key: str) -> str:
    """Extrae el nombre de tarea de una key HDF5.

    Las keys tienen forma 'ses-NNN_task-XXXX'. El número de sesión es
    específico del sujeto, pero el nombre de tarea es compartido.
    Ejemplo: 'ses-001_task-s01e02a' -> 'task-s01e02a'.
    """
    parts = key.split("_", 1)
    return parts[1] if len(parts) > 1 else key


def filter_fmri_for_subject(
    algonauts_dir: Path,
    output_dir: Path,
    chunks_dir: Path,
    subject_id: str,
    tracker: dict,
) -> None:
    """
    Carga el fMRI de un sujeto y filtra para incluir solo los chunks procesados.

    Para garantizar sincronización perfecta con los features de estímulo,
    trunca cada chunk de fMRI al número EXACTO de TRs del tensor de estímulo
    correspondiente (leído desde chunk_{idx:03d}.pt).

    NOTA: El número de sesión (ses-NNN) en las keys HDF5 es específico de cada
    sujeto. Por eso se empareja por nombre de tarea (task-XXXX), no por key
    completa.

    Args:
        algonauts_dir: Directorio raíz de Algonauts 2025.
        output_dir: Directorio de salida para tensores filtrados.
        chunks_dir: Directorio con los chunk_{i:03d}.pt extraídos.
        subject_id: ID del sujeto (ej: sub-01).
        tracker: Diccionario del tracker JSON con chunks marcados.
    """
    processed_indices = {i for i, c in enumerate(tracker["chunks"]) if c["processed"]}
    if not processed_indices:
        print(f"   No hay chunks procesados. No se generará fMRI para {subject_id}.")
        return

    print(f"\n{'=' * 60}")
    print(f"Filtrando fMRI para {subject_id}")
    print(f"{'=' * 60}")
    print(f"  Chunks a incluir: {len(processed_indices)}/{len(tracker['chunks'])}")

    # Obtener la lista de keys en el MISMO orden que get_chunks_info
    all_chunks_info = get_chunks_info(str(algonauts_dir))

    # Cargar HDF5 del sujeto y construir mapa: task_name -> (full_key, h5_path)
    paths = get_fmri_paths(algonauts_dir, subject_id)
    task_to_source: dict[str, tuple[str, Path]] = {}

    for task_name, h5_path in paths.items():
        if not h5_path.exists():
            print(f"  {task_name}: Archivo no encontrado, saltando.")
            continue
        with h5py.File(h5_path, "r") as f:
            for key in f.keys():
                task = _task_from_key(key)
                if task in task_to_source:
                    print(f"    ADVERTENCIA: tarea duplicada '{task}' en {task_name}, usando primera ocurrencia.")
                    continue
                task_to_source[task] = (key, h5_path)

    all_filtered_data = []
    total_trs_extracted = 0

    for idx in sorted(processed_indices):
        if idx >= len(all_chunks_info):
            continue

        stimulus_key = all_chunks_info[idx]["key"]
        stimulus_task = _task_from_key(stimulus_key)

        if stimulus_task not in task_to_source:
            print(f"   {stimulus_key}: Tarea '{stimulus_task}' no encontrada en fMRI, saltando.")
            continue

        fmri_key, h5_path = task_to_source[stimulus_task]

        # Cargar el tensor de estímulo para saber cuántos TRs realmente tiene
        stimulus_path = chunks_dir / f"chunk_{idx:03d}.pt"
        if not stimulus_path.exists():
            print(f"    chunk_{idx:03d}.pt no encontrado, saltando.")
            continue
        stimulus_tensor = torch.load(stimulus_path, weights_only=True)
        num_trs_stimulus = stimulus_tensor.shape[0]

        # Extraer fMRI y truncar exactamente al número de TRs del estímulo
        with h5py.File(h5_path, "r") as f:
            chunk_data = f[fmri_key][:].astype(np.float32)

        if chunk_data.ndim == 1:
            chunk_data = chunk_data.reshape(1, -1)
        if chunk_data.shape[1] != NUM_PARCELS:
            if chunk_data.shape[0] == NUM_PARCELS:
                chunk_data = chunk_data.T
            else:
                print(f"  Key '{stimulus_key}' forma inesperada {chunk_data.shape}, saltando.")
                continue

        # TRUNCAR al número de TRs del estímulo (garantiza sincronización)
        if chunk_data.shape[0] > num_trs_stimulus:
            chunk_data = chunk_data[:num_trs_stimulus]
            truncation_msg = f" (truncado de {chunk_data.shape[0] + 1} a {num_trs_stimulus})"
        elif chunk_data.shape[0] < num_trs_stimulus:
            truncation_msg = f"  ADVERTENCIA: fMRI más corto ({chunk_data.shape[0]} < {num_trs_stimulus})"
        else:
            truncation_msg = ""

        all_filtered_data.append(chunk_data)
        total_trs_extracted += chunk_data.shape[0]
        print(f"   {stimulus_key} -> {fmri_key}: {chunk_data.shape[0]} TRs{truncation_msg}")

    if not all_filtered_data:
        print(f" No se encontraron datos para {subject_id}")
        return

    combined = np.concatenate(all_filtered_data, axis=0)
    tensor = torch.tensor(combined, dtype=torch.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{subject_id}.pt"
    torch.save(tensor, out_path)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\n  Guardado: {out_path}")
    print(f"  Forma: {tensor.shape}")
    print(f"  Tamaño: {size_mb:.1f} MB")
    print(f"  TRs extraídos: {total_trs_extracted:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Filtra fMRI de Algonauts para incluir solo chunks con estímulos procesados."
    )
    parser.add_argument(
        "--algonauts_dir",
        type=str,
        default="./algonauts_2025",
        help="Directorio raíz del dataset Algonauts 2025.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/subjects_fmri_filtered",
        help="Directorio de salida para tensores filtrados.",
    )
    parser.add_argument(
        "--chunks_dir",
        type=str,
        default="./data/features/chunks",
        help="Directorio con los chunk_{i:03d}.pt extraídos.",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="./data/features/processed_chunks.json",
        help="Ruta al archivo JSON de tracking de chunks procesados.",
    )
    parser.add_argument(
        "--subject_id",
        type=str,
        default=None,
        help="ID del sujeto a filtrar. Si no se especifica, usa --all_subjects.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["sub-01", "sub-02"],
        help="IDs de sujeto a filtrar (default: sub-01 sub-02).",
    )

    args = parser.parse_args()
    algonauts_dir = Path(args.algonauts_dir)
    output_dir = Path(args.output_dir)
    chunks_dir = Path(args.chunks_dir)
    tracker_path = Path(args.tracker)

    if not tracker_path.exists():
        print(f" No se encontró el tracker: {tracker_path}")
        print("   Ejecuta primero la extracción de estímulos para generar el tracker.")
        return

    with open(tracker_path, 'r') as f:
        tracker = json.load(f)

    processed_count = tracker["processed_count"]
    total_count = tracker["total_chunks"]
    print(f"Tracker cargado: {processed_count}/{total_count} chunks procesados")

    if processed_count == 0:
        print(" No hay chunks procesados en el tracker.")
        return

    subjects = [args.subject_id] if args.subject_id else args.subjects

    for sid in subjects:
        try:
            filter_fmri_for_subject(algonauts_dir, output_dir, chunks_dir, sid, tracker)
        except Exception as e:
            print(f"\n  Error filtrando {sid}: {e}")

    print(f"\n{'=' * 60}")
    print("Filtrado completo")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
