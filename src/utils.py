"""
================================================================================
utils.py — Utilidades auxiliares del pipeline
================================================================================

Este módulo contiene funciones de utilidad que no encajan en otros módulos.
Actualmente solo tiene extracción de audio desde archivos de video.

DEPENDENCIAS
------------
• ffmpeg (sistema) — decodificación de MKV a WAV
• soundfile — lectura de audio desde bytes en memoria
• librosa — fallback para carga de audio

NOTA: ffmpeg debe estar instalado en el sistema operativo.
      En macOS: brew install ffmpeg
================================================================================
"""

import subprocess
import tempfile
from pathlib import Path
import librosa
import numpy as np
import io
import soundfile as sf

def extract_audio_from_mkv(mkv_path: Path, sr: int = 16000) -> np.ndarray:
    """
    Extrae el audio de un archivo .mkv usando ffmpeg.
    
    librosa no puede leer .mkv directamente, así que usamos ffmpeg para 
    convertir a WAV en memoria (stdout pipe), y luego lo cargamos con librosa.
    
    Args:
        mkv_path: Ruta al archivo .mkv.
        sr: Sample rate deseado (16kHz para Gemma 4).
    
    Returns:
        numpy array con la forma de onda del audio (mono, float32).
    """
    try:
        # ffmpeg convierte MKV → WAV mono 16kHz directo a stdout
        cmd = [
            "ffmpeg", "-i", str(mkv_path),
            "-vn",                    # Sin video
            "-acodec", "pcm_s16le",   # WAV PCM 16-bit
            "-ar", str(sr),           # Sample rate
            "-ac", "1",               # Mono
            "-f", "wav",              # Formato de salida
            "-loglevel", "error",     # Solo errores
            "pipe:1"                  # Stdout
        ]
        result = subprocess.run(cmd, capture_output=True, check=True)
        
        # Cargar el WAV desde los bytes en memoria
        audio_data, _ = sf.read(io.BytesIO(result.stdout), dtype='float32')
        
        return audio_data
        
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"    ffmpeg falló para {mkv_path.name}: {e}")
        print("     Intentando fallback con librosa...")
        try:
            # Fallback: escribir a archivo temporal
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                subprocess.run([
                    "ffmpeg", "-i", str(mkv_path),
                    "-vn", "-acodec", "pcm_s16le",
                    "-ar", str(sr), "-ac", "1",
                    "-loglevel", "error", "-y",
                    tmp.name
                ], check=True)
                audio, _ = librosa.load(tmp.name, sr=sr, mono=True)
                return audio
        except Exception as e2:
            print(f"  No se pudo extraer audio: {e2}")
            return np.zeros(1)  # Array mínimo, se expandirá después
