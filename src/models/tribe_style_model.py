"""
tribe_style_model.py — Modelo Brain Encoding alineado con la arquitectura TriBE.

Arquitectura:
    ViT features    (B, W, D_vit)  ──► FFN_vit  ──► LayerNorm ──► (B, W, 512)
    Conformer feats (B, W, D_conf) ──► FFN_conf ──► LayerNorm ──► (B, W, 512)  ──► cat ──► (B, W, 1536)
    Text features   (B, W, D_text) ──► FFN_text ──► LayerNorm ──► (B, W, 512)

    ──► Transformer Encoder (4L, 8H) sobre (B, W, 1536)
    ──► Subject-Conditional Linear: (B, W, 1536) ──► (B, W, num_vertices)

Diferencias clave respecto al modelo v2 (TemporalFullModel):
    1. 3 FFNs independientes en vez de un único bottleneck
       → Cada modalidad mantiene su "identidad" antes de la fusión
    2. Subject-Conditional head (un Linear por sujeto)
       → Adapta la predicción a la anatomía de cada cerebro
    3. Input: 3 tensores separados (vit, conformer, text) en vez de uno fusionado
       → Permite ablaciones limpias por modalidad

Ablaciones soportadas:
    "full"          → vit + conformer + text (modelo completo)
    "vit_only"      → solo ViT, ceros para conformer y texto
    "conformer_only"→ solo Conformer
    "text_only"     → solo texto (equivalente semántico al baseline v2)
    "vit_conformer" → ViT + Conformer sin texto (features sensoriales puros)
    "ridge_full"    → concatenación directa → regresión lineal (ver ridge_model.py)
"""

import torch
import torch.nn as nn

# pyrefly: ignore [missing-import]
from src.models.base_module import BrainEncodingModule
# pyrefly: ignore [missing-import]
from src.config import ModelConfig
# pyrefly: ignore [missing-import]
from src.architecture.temporal_transformer import TemporalTransformerEncoder


class ModalityFFN(nn.Module):
    """
    FFN de proyección para una única modalidad.

    Proyecta de input_dim → shared_dim con LayerNorm y activación GELU.
    Equivalente al 'linear layer with shared output dimension' de TriBE.
    """

    def __init__(self, input_dim: int, shared_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TriBEStyleModel(BrainEncodingModule):
    """
    Modelo brain encoding con 3 FFNs independientes + Transformer + Subject head.

    Args:
        window_size:         TRs por ventana (default 67 ≈ 100s).
        transformer_layers:  Capas del Transformer encoder.
        transformer_heads:   Cabezas de atención.
        transformer_dropout: Dropout en el Transformer.
        modality_dropout:    Dropout en las FFNs de modalidad.
        ablation:            Modalidades activas. Ver docstring del módulo.
        num_subjects:        Número de sujetos (para subject-conditional head).
        **kwargs:            Parámetros de BrainEncodingModule (lr, etc.).
    """

    VALID_ABLATIONS = {
        "full", "vit_only", "conformer_only", "text_only", "vit_conformer",
    }

    def __init__(
        self,
        window_size: int = 67,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        transformer_dropout: float = 0.2,
        modality_dropout: float = 0.1,
        ablation: str = "full",
        num_subjects: int = 2,
        **kwargs,
    ):
        super().__init__(model_name=f"tribe_{ablation}", **kwargs)

        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(
                f"ablation='{ablation}' no válido. Opciones: {self.VALID_ABLATIONS}"
            )

        self.window_size = window_size
        self.ablation    = ablation
        self.config      = ModelConfig()
        shared_dim       = self.config.shared_dim  # 512

        # ── FFNs independientes por modalidad ─────────────────────────────
        self.vit_ffn = ModalityFFN(
            self.config.vit_hidden_size, shared_dim, modality_dropout
        )
        self.conformer_ffn = ModalityFFN(
            self.config.conformer_hidden_size, shared_dim, modality_dropout
        )
        self.text_ffn = ModalityFFN(
            self.config.gemma_hidden_size, shared_dim, modality_dropout
        )

        # ── Transformer Encoder sobre la concatenación [3 × shared_dim] ──
        transformer_dim = self.config.transformer_dim  # shared_dim * 3 = 1536
        self.transformer = TemporalTransformerEncoder(
            d_model=transformer_dim,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            max_window=window_size * 2,
            dropout=transformer_dropout,
            num_subjects=0,  # El subject embedding se maneja en el head
        )

        self.dropout = nn.Dropout(0.2)

        # ── Subject-Conditional Linear (un head por sujeto) ───────────────
        # Siguiendo TriBE: cada sujeto tiene sus propios pesos de salida.
        # Esto permite que el modelo adapte la predicción a la anatomía
        # individual de cada cerebro.
        self.num_subjects = num_subjects
        if num_subjects > 1:
            self.subject_heads = nn.ModuleList([
                nn.Linear(transformer_dim, self.hparams.num_vertices)
                for _ in range(num_subjects)
            ])
        else:
            # Un único head si solo hay un sujeto
            self.subject_heads = nn.ModuleList([
                nn.Linear(transformer_dim, self.hparams.num_vertices)
            ])

        # Mapeo de subject_id string → índice
        self._subject_map: dict[str, int] = {}

    def _get_subject_idx(self, subject_id: str) -> int:
        """Convierte sub-01 → 0, sub-02 → 1, etc."""
        if subject_id not in self._subject_map:
            # Extraer número y usar como índice (0-based)
            import re
            match = re.search(r"(\d+)", subject_id)
            idx = (int(match.group(1)) - 1) % self.num_subjects if match else 0
            self._subject_map[subject_id] = idx
        return self._subject_map[subject_id]

    def _apply_ablation(
        self,
        vit: torch.Tensor,
        conformer: torch.Tensor,
        text: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Aplica la ablación zeroing out las modalidades inactivas.
        Esto permite comparaciones justas: la arquitectura es idéntica,
        solo cambia qué señal fluye.
        """
        if self.ablation == "vit_only":
            conformer = torch.zeros_like(conformer)
            text      = torch.zeros_like(text)
        elif self.ablation == "conformer_only":
            vit  = torch.zeros_like(vit)
            text = torch.zeros_like(text)
        elif self.ablation == "text_only":
            vit      = torch.zeros_like(vit)
            conformer = torch.zeros_like(conformer)
        elif self.ablation == "vit_conformer":
            text = torch.zeros_like(text)
        # "full" no hace nada
        return vit, conformer, text

    def forward(
        self,
        vit: torch.Tensor,
        conformer: torch.Tensor,
        text: torch.Tensor,
        subject_id: str = "sub-01",
    ) -> torch.Tensor:
        """
        Args:
            vit:        (B, W, D_vit)       — features ViT pre-proyección.
            conformer:  (B, W, D_conformer) — features Conformer pre-proyección.
            text:       (B, W, D_text)      — features decoder LLM.
            subject_id: Identificador del sujeto (ej: "sub-01").

        Returns:
            predicted_bold: (B, W, num_vertices)
        """
        # 1. Ablación (zeroing de modalidades inactivas)
        vit, conformer, text = self._apply_ablation(vit, conformer, text)

        # 2. FFN independiente por modalidad → (B, W, shared_dim) cada una
        v = self.vit_ffn(vit)
        a = self.conformer_ffn(conformer)
        t = self.text_ffn(text)

        # 3. Concatenar → (B, W, 3 × shared_dim = 1536)
        x = torch.cat([v, a, t], dim=-1)

        # 4. Transformer Encoder temporal
        x = self.transformer(x)
        x = self.dropout(x)

        # 5. Subject-Conditional head
        subj_idx = self._get_subject_idx(subject_id)
        head = self.subject_heads[min(subj_idx, len(self.subject_heads) - 1)]

        B, W, D = x.shape
        out = head(x.reshape(B * W, D)).reshape(B, W, self.hparams.num_vertices)
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # Lightning steps (reciben batch como tupla de 4 tensores)
    # ─────────────────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        vit, conformer, text, bold = batch
        pred = self(vit, conformer, text, self.hparams.get("subject_id", "sub-01"))
        loss = nn.functional.mse_loss(pred, bold)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        vit, conformer, text, bold = batch
        pred = self(vit, conformer, text, self.hparams.get("subject_id", "sub-01"))
        loss = nn.functional.mse_loss(pred, bold)

        pred_flat = pred.reshape(-1, self.hparams.num_vertices)
        bold_flat = bold.reshape(-1, self.hparams.num_vertices)
        pearson   = self._compute_pearson(pred_flat, bold_flat)
        avg_p     = pearson.mean()

        self.log("val/loss",    loss,  on_epoch=True, prog_bar=True)
        self.log("val/pearson", avg_p, on_epoch=True, prog_bar=True)
        self.log("val/pearson>0.15",
                 (pearson > 0.15).float().mean(), on_epoch=True)
        self._last_val_pearson_map = pearson.detach().cpu()
        return {"val_loss": loss, "val_pearson": avg_p}

    def test_step(self, batch, batch_idx):
        vit, conformer, text, bold = batch
        pred = self(vit, conformer, text, self.hparams.get("subject_id", "sub-01"))
        loss = nn.functional.mse_loss(pred, bold)

        pred_flat = pred.reshape(-1, self.hparams.num_vertices)
        bold_flat = bold.reshape(-1, self.hparams.num_vertices)
        pearson   = self._compute_pearson(pred_flat, bold_flat)
        avg_p     = pearson.mean()

        self.log("test/loss",    loss,  on_epoch=True)
        self.log("test/pearson", avg_p, on_epoch=True)
        self._last_test_pearson_map = pearson.detach().cpu()
        return {"test_loss": loss, "test_pearson": avg_p}
