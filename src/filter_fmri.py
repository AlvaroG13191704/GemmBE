"""
Filtra datos fMRI de Algonauts 2025 para incluir SOLO los chunks
que fueron procesados por el extractor de estímulos.

Uso:
    # Filtrar todos los sujetos usando el tracker por defecto
    uv run python src/filter_fmri.py --all_subjects

    # Filtrar solo un sujeto
    uv run python src/filter_fmri.py --subject_id sub-01

    # Especificar tracker y output manualmente
    uv run python src/filter_fmri.py --all_subjects \
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

from src.prepare_fmri import (
    ALGONAUTS_SUBJECTS,
    NUM_PARCELS,
    get_fmri_paths,
)
from src.extract_features import get_chunks_info


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

    # Cargar HDF5 del sujeto
    paths = get_fmri_paths(algonauts_dir, subject_id)
    all_filtered_data = []
    total_trs_extracted = 0

    for task_name, h5_path in paths.items():
        if not h5_path.exists():
            print(f"  {task_name}: Archivo no encontrado, saltando.")
            continue

        with h5py.File(h5_path, 'r') as f:
            for idx in sorted(processed_indices):
                if idx >= len(all_chunks_info):
                    continue
                key = all_chunks_info[idx]["key"]
                if key not in f:
                    continue

                # Cargar el tensor de estímulo para saber cuántos TRs realmente tiene
                stimulus_path = chunks_dir / f"chunk_{idx:03d}.pt"
                if not stimulus_path.exists():
                    print(f"    chunk_{idx:03d}.pt no encontrado, saltando.")
                    continue
                stimulus_tensor = torch.load(stimulus_path, weights_only=True)
                num_trs_stimulus = stimulus_tensor.shape[0]

                # Extraer fMRI y truncar exactamente al número de TRs del estímulo
                chunk_data = f[key][:].astype(np.float32)
                if chunk_data.ndim == 1:
                    chunk_data = chunk_data.reshape(1, -1)
                if chunk_data.shape[1] != NUM_PARCELS:
                    if chunk_data.shape[0] == NUM_PARCELS:
                        chunk_data = chunk_data.T
                    else:
                        print(f"  Key '{key}' forma inesperada {chunk_data.shape}, saltando.")
                        continue

                # TRUNCAR al número de TRs del estímulo (garantiza sincronización)
                if chunk_data.shape[0] > num_trs_stimulus:
                    chunk_data = chunk_data[:num_trs_stimulus]
                    truncation_msg = f" (truncado de {f[key].shape[0]} a {num_trs_stimulus})"
                elif chunk_data.shape[0] < num_trs_stimulus:
                    # Esto no debería pasar, pero lo reportamos
                    truncation_msg = f"  fMRI más corto que estímulo! ({chunk_data.shape[0]} < {num_trs_stimulus})"
                else:
                    truncation_msg = ""

                all_filtered_data.append(chunk_data)
                total_trs_extracted += chunk_data.shape[0]
                print(f"   {key}: {chunk_data.shape[0]} TRs{truncation_msg}")

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
        "--all_subjects",
        action="store_true",
        help="Filtrar todos los sujetos.",
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
    print(f"📋 Tracker cargado: {processed_count}/{total_count} chunks procesados")

    if processed_count == 0:
        print(" No hay chunks procesados en el tracker.")
        return

    subjects = [args.subject_id] if args.subject_id else ALGONAUTS_SUBJECTS

    if not args.all_subjects and not args.subject_id:
        print("Error: Especifica --subject_id o --all_subjects.")
        return

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
