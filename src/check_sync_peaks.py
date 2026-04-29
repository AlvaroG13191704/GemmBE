import torch
import cv2
import librosa
import numpy as np
import argparse
from pathlib import Path

def check_sync_peaks(video_path: str, bold_path: str, tr_duration: float = 1.5):
    print("\n" + "=" * 60)
    print("🔍 DIAGNÓSTICO DE SINCRONIZACIÓN (Audio vs Cerebro)")
    print("=" * 60)

    # 1. Cargar Audio del Video y calcular Energía
    print(f"  🔊 Extrayendo energía de audio de: {video_path}")
    audio_waveform, sr = librosa.load(video_path, sr=16000)
    
    # Calcular energía (RMS) por cada ventana de 1.5s (TR)
    samples_per_tr = int(tr_duration * sr)
    num_trs_audio = len(audio_waveform) // samples_per_tr
    
    audio_energy = []
    for i in range(num_trs_audio):
        segment = audio_waveform[i*samples_per_tr : (i+1)*samples_per_tr]
        rms = np.sqrt(np.mean(segment**2))
        audio_energy.append(rms)
    
    audio_energy = np.array(audio_energy)
    # Normalizar para comparar
    audio_energy = (audio_energy - audio_energy.mean()) / audio_energy.std()
    print(f"  ✅ Energía de audio calculada para {len(audio_energy)} TRs.")

    # 2. Cargar fMRI y buscar el vóxel más "auditivo"
    print(f"  🧠 Cargando BOLD de: {bold_path}")
    bold = torch.load(bold_path, weights_only=True).float().numpy()
    
    # En lugar de señal global (que es ruido), buscamos vóxeles que reaccionen al audio
    # para confirmar que el tiempo es correcto.
    print("  🔍 Buscando vóxeles con respuesta auditiva (esto puede tardar unos segundos)...")
    
    # Recortamos para alinear
    lag_test = 3 # 4.5s
    min_len = min(len(audio_energy), len(bold)) - lag_test
    
    # Calculamos correlación de CADA vóxel con el audio en el lag biológico (3 TRs)
    audio_ref = audio_energy[0:min_len]
    bold_ref = bold[lag_test : lag_test + min_len]
    
    # Correlación masiva: (N_voxels,)
    def get_correlations(target, matrix):
        target = (target - target.mean()) / (target.std() + 1e-8)
        matrix = (matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + 1e-8)
        return (target[:, None] * matrix).mean(axis=0)
    
    voxel_corrs = get_correlations(audio_ref, bold_ref)
    best_voxel_idx = np.argmax(voxel_corrs)
    max_corr_found = voxel_corrs[best_voxel_idx]
    
    print(f"  ✅ Vóxel más auditivo encontrado: Vóxel #{best_voxel_idx} (r={max_corr_found:.4f})")
    
    # Usamos ese vóxel como nuestra señal de referencia
    reference_signal = bold[:, best_voxel_idx]
    # Normalizar
    reference_signal = (reference_signal - reference_signal.mean()) / reference_signal.std()

    # 3. Análisis de Correlación Cruzada sobre el mejor vóxel
    print("\n📊 ANALIZANDO CORRELACIÓN POR LAG (Usando el vóxel #{}):".format(best_voxel_idx))
    
    max_lag = 10
    lags = range(max_lag + 1)
    correlations = []
    
    min_len = min(len(audio_energy), len(reference_signal)) - max_lag
    
    for lag in lags:
        a = audio_energy[0 : min_len]
        b = reference_signal[lag : lag + min_len]
        corr = np.corrcoef(a, b)[0, 1]
        correlations.append(corr)
        
        star = " ⭐ (ESPERADO)" if lag == 3 or lag == 4 else ""
        print(f"   Lag {lag:>2} TRs ({lag*tr_duration:>4.1f}s): {corr:>8.4f}{star}")

    # 4. Veredicto
    best_lag = np.argmax(correlations)
    print("\n" + "─" * 60)
    print(f"🏆 RESULTADO: El pico de correlación está en el Lag {best_lag} ({best_lag*tr_duration:.1f}s)")
    
    if 2 <= best_lag <= 5:
        print("✅ VEREDICTO: ¡Sincronización Exitosa! El cerebro reacciona con el retraso biológico esperado (4-6s).")
    else:
        print("❌ VEREDICTO: Problema de sincronización. El pico no está en el rango biológico (4-6s).")
    print("─" * 60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="dataset/stimuli_Sherlock.m4v")
    parser.add_argument("--bold", type=str, default="data/features/fmri/sub-01.pt")
    parser.add_argument("--tr", type=float, default=1.5)
    args = parser.parse_args()
    
    check_sync_peaks(args.video, args.bold, args.tr)
