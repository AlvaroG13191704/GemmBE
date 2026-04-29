"""
Preparación de datos fMRI: HDF5 → Tensores PyTorch con reducción de vóxeles.

Este script convierte los archivos HDF5 preprocesados del dataset Sherlock
(generados por nltools/fmriprep) en tensores .pt optimizados para entrenamiento.

=== REDUCCIÓN DE VÓXELES (BRAIN MASKING) ===

El HDF5 original contiene 238,955 vóxeles (máscara cerebral completa MNI152).
Sin embargo, ~15-20% de esos vóxeles son sustancia blanca, líquido cefalorraquídeo
o áreas con señal BOLD negligible que solo añaden ruido y parámetros al modelo.

Estrategia de reducción:
    1. Calcular la varianza temporal de cada vóxel a lo largo de todos los TRs.
    2. Los vóxeles con baja varianza temporal son "muertos" — no responden a
       estímulos y no contribuyen información útil al modelo.
    3. Eliminar el percentil inferior (configurable, default: 20%).
    4. Guardar una máscara binaria (.pt) para reconstruir el volumen 3D después.

Impacto con --variance_threshold 20 (default):
    238,955 → ~191,000 vóxeles (reducción del 20%)
    SubjectBlock: 122M → ~98M parámetros (-20%)
    BOLD por sujeto: 862 MB → 690 MB (-172 MB)
    
    En MacBook M4 24GB esto permite entrenar ~7 sujetos en vez de ~5.

Uso:
    # Preparar un sujeto con reducción de vóxeles (recomendado)
    uv run python src/prepare_fmri.py \\
        --hdf5_path Sherlock/fmriprep/sub-01/func/sub-01_..._bold.hdf5 \\
        --subject_id sub-01

    # Sin reducción (mantener 238,955 vóxeles)
    uv run python src/prepare_fmri.py \\
        --hdf5_path ... --subject_id sub-01 --no_reduce

    # Reducción más agresiva (quitar bottom 30%)
    uv run python src/prepare_fmri.py \\
        --hdf5_path ... --subject_id sub-01 --variance_threshold 30
"""

import argparse
import os
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import torch


def compute_variance_mask(data: np.ndarray, threshold_pct: float = 20.0) -> np.ndarray:
    """
    Calcula una máscara binaria basada en la varianza temporal de cada vóxel.
    
    Los vóxeles con baja varianza temporal son "muertos": sustancia blanca,
    líquido cefalorraquídeo, o vóxeles fuera del parénquima cerebral que
    no responden a estímulos.
    
    Args:
        data: (num_trs, num_voxels) — serie temporal BOLD completa.
        threshold_pct: Percentil inferior a eliminar (0-100).
                      20 = quitar el 20% con menor varianza.
    
    Returns:
        mask: (num_voxels,) — bool array. True = vóxel activo (mantener).
    """
    # Varianza temporal por vóxel
    temporal_var = data.var(axis=0)
    
    # Umbral: percentil inferior
    cutoff = np.percentile(temporal_var, threshold_pct)
    
    # Máscara: True para vóxeles con varianza > umbral
    mask = temporal_var > cutoff
    
    return mask


def prepare_fmri(
    hdf5_path: str,
    output_dir: str,
    subject_id: str,
    variance_threshold: float = 20.0,
    reduce: bool = True,
):
    """
    Convierte un HDF5 de Sherlock en tensores PyTorch optimizados.
    
    Si reduce=True (default), aplica una máscara de varianza temporal
    para eliminar vóxeles inactivos y reducir el tamaño del modelo.
    
    La máscara se genera POR SUJETO (cada cerebro es diferente) pero
    se puede compartir una máscara común si se desea consistencia.
    
    Args:
        hdf5_path: Ruta al archivo .hdf5 preprocesado.
        output_dir: Directorio donde guardar los tensores.
        subject_id: ID del sujeto (e.g., "sub-01").
        variance_threshold: Percentil de varianza a eliminar (0-100).
        reduce: Si True, aplica reducción de vóxeles.
    """
    print("=" * 60)
    print(f"🧠 Preparando fMRI para {subject_id}")
    print("=" * 60)
    
    # 1. Leer el HDF5
    print(f"  📂 Leyendo: {hdf5_path}")
    with h5py.File(hdf5_path, 'r') as f:
        data = f['data'][:]  # (num_trs, 238955)
        
        # Guardar metadatos del espacio MNI para reconstrucción futura
        mask_affine = f['mask_affine'][:] if 'mask_affine' in f else None
        
    print(f"  ✅ Datos cargados. Forma original: {data.shape}")
    print(f"     TRs: {data.shape[0]} | Vóxeles: {data.shape[1]:,}")
    
    # 2. Reducción de vóxeles (opcional)
    if reduce and variance_threshold > 0:
        print(f"\n  🔬 Aplicando reducción de vóxeles (umbral: bottom {variance_threshold}%)...")
        
        voxel_mask = compute_variance_mask(data, variance_threshold)
        original_count = data.shape[1]
        active_count = voxel_mask.sum()
        removed_count = original_count - active_count
        
        print(f"     Vóxeles originales:  {original_count:,}")
        print(f"     Vóxeles activos:     {active_count:,}")
        print(f"     Vóxeles eliminados:  {removed_count:,} ({100*removed_count/original_count:.1f}%)")
        
        # Aplicar máscara
        data_reduced = data[:, voxel_mask]
        print(f"     Forma reducida: {data_reduced.shape}")
        
        # Estadísticas de calidad
        temporal_var = data.var(axis=0)
        var_kept = temporal_var[voxel_mask]
        var_removed = temporal_var[~voxel_mask]
        print(f"     Varianza media (mantenidos): {var_kept.mean():.4f}")
        print(f"     Varianza media (eliminados): {var_removed.mean():.6f}")
    else:
        print("\n  ⏭️  Sin reducción de vóxeles (--no_reduce)")
        data_reduced = data
        voxel_mask = np.ones(data.shape[1], dtype=bool)
    
    # 3. Convertir a Tensor PyTorch
    tensor = torch.tensor(data_reduced, dtype=torch.float32)
    mask_tensor = torch.tensor(voxel_mask, dtype=torch.bool)
    
    # 4. Guardar tensores
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # BOLD data
    bold_path = out_dir / f"{subject_id}.pt"
    torch.save(tensor, bold_path)
    
    # Máscara de vóxeles (compartida para reconstrucción)
    mask_path = out_dir / "voxel_mask.pt"
    if not mask_path.exists() or reduce:
        mask_info = {
            "mask": mask_tensor,                      # (238955,) bool
            "original_num_voxels": data.shape[1],     # 238955
            "active_num_voxels": int(voxel_mask.sum()),
            "variance_threshold_pct": variance_threshold,
            "mask_affine": torch.tensor(mask_affine, dtype=torch.float64) if mask_affine is not None else None,
            "volume_shape": (91, 109, 91),            # MNI152 2mm grid
        }
        torch.save(mask_info, mask_path)
        print(f"\n  🎭 Máscara guardada en: {mask_path}")
    
    # Reporte final
    bold_size_mb = os.path.getsize(bold_path) / (1024 * 1024)
    print(f"\n  💾 BOLD guardado en: {bold_path}")
    print(f"     Forma: {tensor.shape}")
    print(f"     Dtype: {tensor.dtype}")
    print(f"     Tamaño: {bold_size_mb:.1f} MB")
    
    # Impacto en el modelo
    num_voxels = tensor.shape[1]
    sb_params = 512 * num_voxels + num_voxels
    sb_train_mb = sb_params * 16 / (1024**2)
    print(f"\n  📊 Impacto en SubjectBlock:")
    print(f"     Parámetros: {sb_params:,}")
    print(f"     RAM (train): {sb_train_mb:.0f} MB")


def prepare_all_subjects(
    sherlock_dir: str,
    output_dir: str,
    variance_threshold: float = 20.0,
    reduce: bool = True,
    part: str = "Part1",
):
    """
    Prepara todos los sujetos encontrados en el directorio Sherlock.
    
    Busca archivos HDF5 que sigan el patrón del dataset Sherlock
    y los procesa uno por uno.
    
    Args:
        sherlock_dir: Directorio raíz de Sherlock (contiene fmriprep/).
        output_dir: Directorio donde guardar los tensores.
        variance_threshold: Percentil de varianza a eliminar.
        reduce: Si True, aplica reducción de vóxeles.
        part: "Part1" o "Part2" del experimento Sherlock.
    """
    fmriprep_dir = Path(sherlock_dir) / "fmriprep"
    
    # Encontrar todos los sujetos
    subject_dirs = sorted([
        d for d in fmriprep_dir.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    ])
    
    print(f"\n🔍 Encontrados {len(subject_dirs)} sujetos en {fmriprep_dir}")
    
    processed = 0
    skipped = 0
    
    for sub_dir in subject_dirs:
        subject_id = sub_dir.name
        func_dir = sub_dir / "func"
        
        # Buscar el HDF5 correspondiente
        pattern = f"{subject_id}_denoise_crop_smooth6mm_task-sherlock{part}_space-MNI152NLin2009cAsym_desc-preproc_bold.hdf5"
        hdf5_path = func_dir / pattern
        
        if not hdf5_path.exists():
            # Check if it's a broken symlink (datalad placeholder)
            if hdf5_path.is_symlink():
                print(f"\n  ⚠️  {subject_id}: HDF5 es un placeholder de datalad (no descargado). Saltando.")
            else:
                print(f"\n  ⚠️  {subject_id}: No se encontró {pattern}. Saltando.")
            skipped += 1
            continue
        
        try:
            prepare_fmri(
                hdf5_path=str(hdf5_path),
                output_dir=output_dir,
                subject_id=subject_id,
                variance_threshold=variance_threshold,
                reduce=reduce,
            )
            processed += 1
        except Exception as e:
            print(f"\n  ❌ Error procesando {subject_id}: {e}")
            skipped += 1
    
    print(f"\n{'=' * 60}")
    print(f"📊 Resumen: {processed} procesados, {skipped} saltados")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepara datos fMRI de Sherlock para MicroTRIBE-Gemma"
    )
    parser.add_argument(
        "--hdf5_path",
        type=str,
        default=None,
        help="Ruta al archivo .hdf5 (para un solo sujeto).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/features/fmri",
        help="Directorio de salida para tensores .pt (default: ./data/features/fmri).",
    )
    parser.add_argument(
        "--subject_id",
        type=str,
        default="sub-01",
        help="ID del sujeto (default: sub-01).",
    )
    parser.add_argument(
        "--variance_threshold",
        type=float,
        default=20.0,
        help="Percentil de varianza inferior a eliminar (default: 20 = quitar 20%% con menos varianza).",
    )
    parser.add_argument(
        "--no_reduce",
        action="store_true",
        help="Desactivar reducción de vóxeles (mantener 238,955).",
    )
    parser.add_argument(
        "--all_subjects",
        action="store_true",
        help="Procesar todos los sujetos en el directorio Sherlock.",
    )
    parser.add_argument(
        "--sherlock_dir",
        type=str,
        default="./Sherlock",
        help="Directorio raíz de Sherlock (default: ./Sherlock).",
    )
    parser.add_argument(
        "--part",
        type=str,
        default="Part1",
        choices=["Part1", "Part2"],
        help="Parte del experimento Sherlock (default: Part1).",
    )
    
    args = parser.parse_args()
    
    if args.all_subjects:
        prepare_all_subjects(
            sherlock_dir=args.sherlock_dir,
            output_dir=args.output_dir,
            variance_threshold=args.variance_threshold,
            reduce=not args.no_reduce,
            part=args.part,
        )
    elif args.hdf5_path:
        prepare_fmri(
            hdf5_path=args.hdf5_path,
            output_dir=args.output_dir,
            subject_id=args.subject_id,
            variance_threshold=args.variance_threshold,
            reduce=not args.no_reduce,
        )
    else:
        print("Error: Especifica --hdf5_path (un sujeto) o --all_subjects.")
        print("Ejemplo: uv run python src/prepare_fmri.py --hdf5_path path/to/bold.hdf5 --subject_id sub-01")
