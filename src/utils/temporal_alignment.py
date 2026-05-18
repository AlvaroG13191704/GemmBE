"""
Alineación temporal entre la secuencia de tokens de Gemma 4 y la señal fMRI.

Este módulo resuelve dos de los tres retos principales del paper TriBE v2:

Reto 1 — Token → Tiempo:
    Gemma 4 produce una secuencia de tokens de longitud variable (parches de imagen,
    fragmentos de audio, palabras). La fMRI tiene un reloj fijo de ~1 Hz (1 muestra/s).
    TemporalPooling colapsa los tokens en un vector por ventana temporal.

Reto 3 — Desfase Hemodinámico (Causal → Bidireccional):
    TriBE v2 usaba un Encoder (bidireccional) que podía "ver el futuro". Gemma 4
    es un Decoder (causal, solo ve el pasado). HRFAligner compensa esto desplazando
    la ventana de estímulo 5 segundos hacia atrás.
"""

import torch
import torch.nn as nn


class TemporalPooling(nn.Module):
    """
    Colapsa una secuencia de tokens de longitud variable en un vector fijo.
    
    En TriBE v2, se usaba Adaptive Average Pooling para bajar de 2Hz a 1Hz.
    Aquí aplicamos la misma idea pero sobre la dimensión de secuencia de tokens.
    
    Soporta dos modos:
    
    1. 'mean': Promedio simple sobre la secuencia de tokens.
       Equivalente al paso C del main.py original.
       Forma: (B, SeqLen, D) → (B, D)
    
    Args:
        hidden_size: Dimensión del hidden state (1536 para Gemma 4 E2B).
        dtype: Tipo de dato numérico (float16, bfloat16, float32).
    """
    
    def __init__(
        self,
        hidden_size: int = 1536,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
    
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            hidden_states: (Batch, SeqLen, HiddenSize) — salida del decoder de Gemma 4.
            attention_mask: (Batch, SeqLen) — máscara de atención. 1 = token real, 0 = padding.
        
        Returns:
            pooled: (Batch, HiddenSize) — un vector por muestra del batch.
        """
        if attention_mask is not None:
            # Enmascarar tokens de padding antes del promedio
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # (B, S, 1)
            summed = (hidden_states * mask).sum(dim=1)                   # (B, D)
            counts = mask.sum(dim=1).clamp(min=1)                        # (B, 1)
            return summed / counts
        else:
            return hidden_states.mean(dim=1)


class HRFAligner:
    """
    Alineador de Respuesta Hemodinámica (HRF — Hemodynamic Response Function).
    
    El problema:
        Si quieres predecir la actividad cerebral en el segundo T=10,
        necesitas inyectar a Gemma 4 el estímulo de ANTES, porque la sangre
        tarda ~5 segundos en responder al estímulo neuronal.
    
    La solución:
        Para predecir fMRI[T], usamos estímulo[T - hrf_delay].
        
        Ejemplo con hrf_delay=5:
            fMRI[10] ← estímulo[5]
            fMRI[11] ← estímulo[6]
            fMRI[12] ← estímulo[7]
    
    Nota: Esta clase es una utilidad estática para el DataLoader.
    No es un nn.Module porque no tiene parámetros entrenables.
    Se usará cuando se construya el Dataset de Cam-CAN.
    
    Args:
        hrf_delay_seconds: Retraso hemodinámico en segundos (default=5.0).
        fmri_tr_seconds: Tiempo de repetición de la fMRI en segundos (default=1.0).
    """
    
    def __init__(self, hrf_delay_seconds: float = 5.0, fmri_tr_seconds: float = 1.0):
        self.hrf_delay_seconds = hrf_delay_seconds
        self.fmri_tr_seconds = fmri_tr_seconds
        self.delay_in_trs = int(hrf_delay_seconds / fmri_tr_seconds)
    
    def get_stimulus_index(self, fmri_tr_index: int) -> int:
        """
        Dado un índice de TR de fMRI, devuelve el índice del estímulo correspondiente.
        
        Args:
            fmri_tr_index: Índice del TR de fMRI que quieres predecir (0-indexed).
        
        Returns:
            stimulus_index: Índice del estímulo que causó esa activación cerebral.
                           Puede ser negativo si el TR es menor que el delay (esos TRs se descartan).
        """
        return fmri_tr_index - self.delay_in_trs
    
    def get_valid_range(self, total_fmri_trs: int) -> tuple[int, int]:
        """
        Calcula el rango válido de TRs de fMRI para los que hay estímulo disponible.
        
        Los primeros `delay_in_trs` TRs de fMRI no tienen estímulo correspondiente
        (el cerebro aún no ha respondido al inicio de la película).
        
        Args:
            total_fmri_trs: Número total de TRs en el escaneo fMRI.
        
        Returns:
            (start_tr, end_tr): Rango válido (inclusive, exclusive) de TRs para entrenamiento.
        
        Ejemplo:
            Si total_trs=100 y delay=5:
            → (5, 100): Los TRs 0-4 se descartan, se usan TRs 5-99.
        """
        return (self.delay_in_trs, total_fmri_trs)
    
    def align_stimulus_to_fmri(
        self,
        stimulus_features: torch.Tensor,
        fmri_bold: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Alinea temporalmente los features de estímulo con las señales BOLD.
        
        Recorta ambos tensores para que estén correctamente desfasados.
        
        Args:
            stimulus_features: (T_stim, D) — features del estímulo por segundo.
            fmri_bold: (T_fmri, V) — señal BOLD por TR, V=1000 pacelas.
        
        Returns:
            (aligned_stimulus, aligned_bold): Tensores recortados y alineados.
            aligned_stimulus[i] corresponde causalmente a aligned_bold[i].
        """
        # El estímulo que causa fMRI[t] es stimulus[t - delay]
        # Entonces para fMRI[delay:], usamos stimulus[0:T_fmri-delay]
        valid_fmri = fmri_bold[self.delay_in_trs:]                    # (T_fmri - delay, V)
        valid_stim = stimulus_features[:valid_fmri.shape[0]]          # (T_fmri - delay, D)
        
        # Asegurar que no excedemos el estímulo disponible
        min_len = min(valid_fmri.shape[0], valid_stim.shape[0])
        return valid_stim[:min_len], valid_fmri[:min_len]
