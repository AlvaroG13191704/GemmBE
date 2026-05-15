"""
Dataset para features pre-extraídos + señales BOLD de fMRI.

Este Dataset carga los tensores que el OfflineExtractor guardó en disco,
aplica la alineación HRF (desfase hemodinámico de 5s), y retorna pares
(feature, bold) listos para entrenar.

Flujo de datos:
    real_stimulus_features.pt  →  (T, 1536)  ← Features pooled de Gemma 4
    sub-XX.pt                  →  (T, 1000) ← Señal BOLD de fMRI
    
    HRFAligner:
        feature[t-5] se empareja con bold[t]
        Los primeros 5 TRs de BOLD se descartan.
    
    Resultado por muestra:
        feature_i: (1536,)  — embedding de Gemma 4 del estímulo
        bold_i:    (1000,) — actividad BOLD correspondiente

Uso:
    dataset = PreExtractedDataset(
        features_path="./data/features/real_stimulus_features.pt",
        bold_path="./data/features/fmri/sub-01.pt",
    )
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for features, bold in dataloader:
        # features: (32, 1536)
        # bold:     (32, 1000)
        ...
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.temporal_alignment import HRFAligner


class PreExtractedDataset(Dataset):
    """
    Dataset de features pre-extraídos para entrenamiento offline.
    
    Carga los tensores de estímulo y BOLD desde disco, aplica
    la alineación HRF, y retorna pares listos para entrenamiento.
    
    Args:
        features_path: Path al tensor de features (num_trs, 1536).
        bold_path: Path al tensor de BOLD de un sujeto (num_trs, 1000).
        hrf_delay: Desfase hemodinámico en segundos (default: 5.0).
        fmri_tr: Tiempo de repetición de la fMRI en segundos (default: 1.0).
        normalize_bold: Si True, normaliza la señal BOLD (z-score por vértice).
    """
    
    def __init__(
        self,
        features_path: str,
        bold_path: str,
        hrf_delay: float = 5.0,
        fmri_tr: float = 1.49,
        normalize_bold: bool = True,
    ):
        # Cargar tensores desde disco
        self.features = torch.load(features_path, weights_only=True)
        self.bold = torch.load(bold_path, weights_only=True)
        
        # Alinear temporalmente con el desfase HRF
        aligner = HRFAligner(hrf_delay_seconds=hrf_delay, fmri_tr_seconds=fmri_tr)
        self.features, self.bold = aligner.align_stimulus_to_fmri(
            self.features, self.bold
        )
        
        # Normalizar BOLD (z-score por vértice)
        if normalize_bold:
            mean = self.bold.mean(dim=0, keepdim=True)
            std = self.bold.std(dim=0, keepdim=True).clamp(min=1e-8)
            self.bold = (self.bold - mean) / std
        
        print("Dataset cargado:")
        print(f"     Features: {self.features.shape}")
        print(f"     BOLD:     {self.bold.shape}")
        print(f"     Muestras: {len(self)}")
    
    def __len__(self) -> int:
        return self.features.shape[0]
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            feature: (1536,) — embedding pooled de Gemma 4.
            bold:    (1000,) — señal BOLD target.
        """
        return self.features[idx], self.bold[idx]


class MultiSubjectDataset(Dataset):
    """
    Dataset que combina datos de múltiples sujetos.
    
    Permite entrenar el modelo con varios sujetos simultáneamente.
    Cada muestra incluye el subject_id para rutear al SubjectBlock correcto.
    
    Args:
        features_path: Path al tensor de features compartido (num_trs, 1536).
        bold_dir: Directorio con archivos sub-XX.pt de BOLD.
        subject_ids: Lista de IDs de sujeto a incluir.
        hrf_delay: Desfase hemodinámico en segundos.
        normalize_bold: Si True, normaliza BOLD por vértice.
    """
    
    def __init__(
        self,
        features_path: str,
        bold_dir: str,
        subject_ids: list[str],
        hrf_delay: float = 5.0,
        normalize_bold: bool = True,
    ):
        self.subject_ids = subject_ids
        bold_dir = Path(bold_dir)
        
        # Cargar features (compartidos)
        shared_features = torch.load(features_path, weights_only=True)
        
        # Cargar y alinear BOLD de cada sujeto
        aligner = HRFAligner(hrf_delay_seconds=hrf_delay)
        
        self.samples = []  # Lista de (feature, bold, subject_id)
        
        for sid in subject_ids:
            bold_path = bold_dir / f"{sid}.pt"
            if not bold_path.exists():
                print(f"BOLD no encontrado para {sid}: {bold_path}")
                continue
            
            bold = torch.load(bold_path, weights_only=True)
            
            # Alinear HRF para este sujeto
            aligned_feat, aligned_bold = aligner.align_stimulus_to_fmri(
                shared_features, bold
            )
            
            # Normalizar BOLD
            if normalize_bold:
                mean = aligned_bold.mean(dim=0, keepdim=True)
                std = aligned_bold.std(dim=0, keepdim=True).clamp(min=1e-8)
                aligned_bold = (aligned_bold - mean) / std
            
            # Añadir cada TR como una muestra individual
            for t in range(aligned_feat.shape[0]):
                self.samples.append((
                    aligned_feat[t],    # (1536,)
                    aligned_bold[t],    # (1000,)
                    sid,                # str
                ))
            
            print(f"  {sid}: {aligned_feat.shape[0]} muestras cargadas")
        
        print("\n  Dataset multi-sujeto:")
        print(f"     Sujetos: {len(subject_ids)}")
        print(f"     Total muestras: {len(self.samples)}")
        print(f"     Muestras/sujeto: ~{len(self.samples) // max(len(subject_ids), 1)}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            feature: (1536,) — embedding pooled.
            bold:    (1000,) — señal BOLD target.
            subject_id: str — ID del sujeto.
        """
        return self.samples[idx]


def collate_multi_subject(batch):
    """
    Función de collate personalizada para MultiSubjectDataset.
    
    Agrupa por subject_id para poder usar el MultiSubjectBlock.
    
    Usage:
        dataloader = DataLoader(
            dataset,
            batch_size=64,
            collate_fn=collate_multi_subject,
        )
        
        for batch in dataloader:
            for subject_id, features, bold in batch:
                outputs = model.tail_forward(features, subject_id)
                loss = criterion(outputs, bold)
    """
    # Agrupar por subject_id
    from collections import defaultdict
    grouped = defaultdict(lambda: ([], []))
    
    for feature, bold, sid in batch:
        grouped[sid][0].append(feature)
        grouped[sid][1].append(bold)
    
    # Stack por sujeto: retorna lista de (subject_id, features_batch, bold_batch)
    result = []
    for sid, (feats, bolds) in grouped.items():
        result.append((
            sid,
            torch.stack(feats),   # (N_sid, 1536)
            torch.stack(bolds),   # (N_sid, 1000)
        ))
    
    return result


class WindowedDataset(Dataset):
    """
    Dataset que retorna ventanas deslizantes de TRs consecutivos.

    En lugar de devolver un solo TR por muestra, devuelve una ventana
    de W TRs contiguos, permitiendo al Temporal Transformer capturar
    dependencias temporales entre timesteps.

    La ventana deslizante avanza con un stride configurable para controlar
    el solapamiento entre ventanas consecutivas.

    Args:
        features_path: Path al tensor de features (num_trs, 1536).
        bold_path: Path al tensor de BOLD de un sujeto (num_trs, 1000).
        window_size: Número de TRs por ventana (default: 67 ≈ 100s).
        stride: Avance en TRs entre ventanas consecutivas (default: 1).
                stride=1 → máximo solapamiento (más muestras de entrenamiento).
                stride=window_size → sin solapamiento.
        hrf_delay: Desfase hemodinámico en segundos (default: 5.0).
        fmri_tr: Tiempo de repetición de la fMRI en segundos.
        normalize_bold: Si True, normaliza la señal BOLD (z-score por vértice).
    """

    def __init__(
        self,
        features_path: str,
        bold_path: str,
        window_size: int = 67,
        stride: int = 1,
        hrf_delay: float = 5.0,
        fmri_tr: float = 1.49,
        normalize_bold: bool = True,
    ):
        # Cargar tensores desde disco
        features = torch.load(features_path, weights_only=True)
        bold = torch.load(bold_path, weights_only=True)

        # Alinear temporalmente con el desfase HRF
        aligner = HRFAligner(hrf_delay_seconds=hrf_delay, fmri_tr_seconds=fmri_tr)
        self.features, self.bold = aligner.align_stimulus_to_fmri(features, bold)

        # Normalizar BOLD (z-score por vértice, calculado GLOBALMENTE)
        # Es crítico normalizar ANTES de crear ventanas para que las estadísticas
        # reflejen toda la serie temporal, no solo la ventana local.
        if normalize_bold:
            mean = self.bold.mean(dim=0, keepdim=True)
            std = self.bold.std(dim=0, keepdim=True).clamp(min=1e-8)
            self.bold = (self.bold - mean) / std

        self.window_size = window_size
        self.stride = stride
        total_trs = self.features.shape[0]

        # Calcular número de ventanas válidas
        if total_trs < window_size:
            # Si no hay suficientes TRs, usar una sola ventana con padding
            self.num_windows = 1
            self._needs_padding = True
        else:
            self.num_windows = (total_trs - window_size) // stride + 1
            self._needs_padding = False

        print("WindowedDataset cargado:")
        print(f"     Features: {self.features.shape}")
        print(f"     BOLD:     {self.bold.shape}")
        print(f"     Ventana:  {window_size} TRs ({window_size * fmri_tr:.1f}s)")
        print(f"     Stride:   {stride} TRs")
        print(f"     Ventanas: {self.num_windows}")

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            features_window: (W, 1536) — ventana de features consecutivos.
            bold_window:     (W, 1000) — ventana de BOLD target correspondiente.
        """
        if self._needs_padding:
            # Padding con zeros si la serie es más corta que la ventana
            T = self.features.shape[0]
            feat_padded = torch.zeros(self.window_size, self.features.shape[1])
            bold_padded = torch.zeros(self.window_size, self.bold.shape[1])
            feat_padded[:T] = self.features
            bold_padded[:T] = self.bold
            return feat_padded, bold_padded

        start = idx * self.stride
        end = start + self.window_size
        return self.features[start:end], self.bold[start:end]

