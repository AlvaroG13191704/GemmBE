"""
Preparación de datos fMRI: Algonauts 2025 (CNeuroMod) → Tensores PyTorch.

Este script convierte los archivos HDF5 del dataset Algonauts 2025 Challenge
en tensores .pt optimizados para entrenamiento con GemmaBE.

=== DATASET ALGONAUTS 2025 ===

    Fuente: Courtois NeuroMod Project (CNeuroMod)
    Formato: Parcelas Schaefer-1000 (atlas-Schaefer18_parcel-1000Par7Net)
    Espacio: MNI152NLin2009cAsym
    TR: 1.49 segundos
    Sujetos: sub-01, sub-02, sub-03, sub-05
    Estímulos:
        - Friends (temporadas 1-6 train, temporada 7 test)
        - movie10: Bourne Supremacy, Wolf of Wall Street, Life, Hidden Figures

    Cada sujeto tiene:
        - 1 archivo HDF5 para Friends (todas las temporadas, ~515 MB)
        - 1 archivo HDF5 para movie10 (todas las películas, ~92 MB)
    
    Dentro del HDF5, cada key es un chunk (ej: "1e01a" = season 1, episode 1, chunk a).
    Los datos por chunk son de forma (num_trs, 1000) — 1000 parcelas corticales.

=== DIMENSIONALIDAD ===

    Algonauts 2025 usa 1,000 parcelas funcionales (Schaefer-1000 atlas).
    Esto reduce drásticamente los parámetros del SubjectBlock:
        - Sin bottleneck: 1,536 x 1,000 = ~1.5M params
        - Con bottleneck:   512 x 1,000 = ~513K params
    Algonauts permite entrenar en hardware accesible con <500 MB de RAM.

Uso:
    # Preparar un sujeto (friends + movie10, solo train split)
    uv run python src/prepare_fmri.py --subject_id sub-01

    # Preparar todos los sujetos
    uv run python src/prepare_fmri.py --all_subjects

    # Incluir datos de test (season 7) — por defecto excluidos
    uv run python src/prepare_fmri.py --all_subjects --include_test
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch


# === CONSTANTES DEL DATASET ===
ALGONAUTS_SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-05"]
ALGONAUTS_TR = 1.49  # segundos
NUM_PARCELS = 1000

# Seasons 1-6 son training, season 7 es test
TRAIN_SEASONS = [1, 2, 3, 4, 5, 6]
TEST_SEASONS = [7]

# Chunks que TriBE v2 excluye (data issues)
EXCLUDED_CHUNKS = {
    ("s05", "e20a"),
    ("s04", "e01a"),
    ("s06", "e03a"),
    ("s04", "e13b"),
    ("s04", "e01b"),
}


def get_fmri_paths(algonauts_dir: Path, subject_id: str) -> dict[str, Path]:
    """
    Retorna los paths a los archivos HDF5 de fMRI para un sujeto.
    
    Returns:
        dict con keys "friends" y "movie10" → Path al .h5
    """
    func_dir = algonauts_dir / "fmri" / subject_id / "func"
    space = "space-MNI152NLin2009cAsym"
    atlas = "atlas-Schaefer18_parcel-1000Par7Net"
    
    return {
        "friends": func_dir / f"{subject_id}_task-friends_{space}_{atlas}_desc-s123456_bold.h5",
        "movie10": func_dir / f"{subject_id}_task-movie10_{space}_{atlas}_bold.h5",
    }


def is_train_chunk(key: str) -> bool:
    """Determina si un chunk del HDF5 pertenece al split de training."""
    # Friends keys: "ses-001_task-s01e02a", etc.
    # Movie10 keys: "ses-001_task-bourne01", "ses-001_task-life03_run-1", etc.
    # Season 7 = test, seasons 1-6 = train, movies = train
    if "task-s" in key:
        # Extraer la temporada: "task-s01e..." -> "01"
        try:
            task_part = key.split("task-s")[1]
            season = int(task_part[:2])
            return season in TRAIN_SEASONS
        except (IndexError, ValueError):
            pass
    return True


def is_excluded_chunk(key: str) -> bool:
    """Verifica si un chunk está en la lista de exclusión de TriBE v2."""
    for season_str, chunk_str in EXCLUDED_CHUNKS:
        # Construir la cadena que debe aparecer en la key: "task-s05e20a"
        target_str = f"task-{season_str}{chunk_str}"
        if target_str in key:
            return True
    return False


def load_algonauts_fmri(
    h5_path: Path,
    include_test: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """
    Carga todos los chunks de un archivo HDF5 de Algonauts 2025.
    
    Args:
        h5_path: Path al archivo .h5.
        include_test: Si True, incluye season 7 (test).
    
    Returns:
        data: (total_trs, 1000) — todos los chunks concatenados.
        chunk_keys: Lista de keys procesadas (para debug).
    """
    all_data = []
    chunk_keys = []
    
    with h5py.File(h5_path, "r") as f:
        available_keys = sorted(f.keys())
        
        for key in available_keys:
            # Filtrar test si no se quiere
            if not include_test and not is_train_chunk(key):
                continue
            
            # Filtrar chunks excluidos
            if is_excluded_chunk(key):
                continue
            
            chunk_data = f[key][:].astype(np.float32)  # (num_trs, 1000)
            
            # Validar forma
            if chunk_data.ndim == 1:
                # Algunos chunks pueden ser (1000,) si solo hay 1 TR
                chunk_data = chunk_data.reshape(1, -1)
            
            if chunk_data.shape[1] != NUM_PARCELS:
                # Transponer si es (1000, num_trs)
                if chunk_data.shape[0] == NUM_PARCELS:
                    chunk_data = chunk_data.T
                else:
                    print(f" Chunk '{key}' tiene forma inesperada: {chunk_data.shape}. Saltando.")
                    continue
            
            all_data.append(chunk_data)
            chunk_keys.append(key)
    
    if not all_data:
        raise ValueError(f"No se encontraron chunks válidos en {h5_path}")
    
    concatenated = np.concatenate(all_data, axis=0)
    return concatenated, chunk_keys


def prepare_subject(
    algonauts_dir: Path,
    output_dir: Path,
    subject_id: str,
    include_test: bool = False,
) -> None:
    """
    Prepara los datos fMRI de un sujeto de Algonauts 2025.
    
    Carga los HDF5 de Friends y movie10, los concatena, y guarda
    un tensor .pt listo para entrenamiento.
    """
    print(f"\n{'=' * 60}")
    print(f"Preparando fMRI para {subject_id} (Algonauts 2025)")
    print(f"{'=' * 60}")
    
    paths = get_fmri_paths(algonauts_dir, subject_id)
    
    all_data = []
    all_keys = []
    total_chunks = 0
    
    for task_name, h5_path in paths.items():
        if not h5_path.exists():
            # Check if symlink exists but target doesn't (not downloaded)
            if h5_path.is_symlink():
                print(f"  {task_name}: Archivo no descargado (placeholder datalad). Saltando.")
                continue
            else:
                print(f"  {task_name}: Archivo no encontrado: {h5_path}. Saltando.")
                continue
        
        print(f"\n  Cargando {task_name}: {h5_path.name}")
        
        data, keys = load_algonauts_fmri(h5_path, include_test=include_test)
        
        print(f"     Chunks: {len(keys)}")
        print(f"     TRs: {data.shape[0]:,}")
        print(f"     Parcelas: {data.shape[1]}")
        print(f"     Forma: {data.shape}")
        
        all_data.append(data)
        all_keys.extend(keys)
        total_chunks += len(keys)
    
    if not all_data:
        print(f"\n  No se encontraron datos para {subject_id}")
        return
    
    # Concatenar todos los tasks
    combined = np.concatenate(all_data, axis=0)
    
    print("\n  Datos combinados:")
    print(f"     Total TRs: {combined.shape[0]:,}")
    print(f"     Parcelas: {combined.shape[1]}")
    print(f"     Chunks procesados: {total_chunks}")
    
    # Convertir a tensor
    tensor = torch.tensor(combined, dtype=torch.float32)
    
    # Guardar
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{subject_id}.pt"
    torch.save(tensor, out_path)
    
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\n Guardado en: {out_path}")
    print(f"     Forma: {tensor.shape}")
    print(f"     Tamaño: {size_mb:.1f} MB")
    
    # Impacto en el modelo
    sb_params = 512 * NUM_PARCELS + NUM_PARCELS
    print("\n  Impacto en SubjectBlock:")
    print(f"     Parámetros: {sb_params:,}")
    print(f"     RAM (train): {sb_params * 16 / (1024**2):.1f} MB")


def prepare_all_subjects(
    algonauts_dir: Path,
    output_dir: Path,
    include_test: bool = False,
) -> None:
    """Prepara todos los sujetos del dataset."""
    print(f"\n Preparando {len(ALGONAUTS_SUBJECTS)} sujetos de Algonauts 2025")
    
    for subject_id in ALGONAUTS_SUBJECTS:
        try:
            prepare_subject(algonauts_dir, output_dir, subject_id, include_test)
        except Exception as e:
            print(f"\n  Error procesando {subject_id}: {e}")
    
    print(f"\n{'=' * 60}")
    print("Todos los sujetos procesados")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepara datos fMRI de Algonauts 2025 para GemmaBE"
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
        default="./data/subjects_fmri",
        help="Directorio de salida para tensores .pt.",
    )
    parser.add_argument(
        "--subject_id",
        type=str,
        default=None,
        help="ID del sujeto (ej: sub-01). Si no se especifica, usa --all_subjects.",
    )
    parser.add_argument(
        "--all_subjects",
        action="store_true",
        help="Procesar todos los sujetos.",
    )
    parser.add_argument(
        "--include_test",
        action="store_true",
        help="Incluir season 7 (test split). Por defecto solo train (seasons 1-6).",
    )
    
    args = parser.parse_args()
    algonauts_dir = Path(args.algonauts_dir)
    output_dir = Path(args.output_dir)
    
    if args.all_subjects:
        prepare_all_subjects(algonauts_dir, output_dir, args.include_test)
    elif args.subject_id:
        prepare_subject(algonauts_dir, output_dir, args.subject_id, args.include_test)
    else:
        print("Error: Especifica --subject_id o --all_subjects.")
        print("Ejemplo: uv run python src/prepare_fmri.py --subject_id sub-01")
        print("         uv run python src/prepare_fmri.py --all_subjects")
