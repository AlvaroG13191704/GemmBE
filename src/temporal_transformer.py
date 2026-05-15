"""
Temporal Transformer Encoder para integración temporal de features.

En nuestro caso, Gemma 4 ya extrajo features densos (1536d) por TR.
El Temporal Transformer permite que cada TR "vea" a sus vecinos
temporales dentro de una ventana (~100 segundos), capturando la
dinámica de la respuesta hemodinámica de forma aprendida.

Flujo:
    Features pre-extraídos (B, W, 1536)     ← ventana de W TRs
        │
        ▼
    ┌────────────────────────────────────────┐
    │  Bottleneck(1536→512) per-timestep    │
    │  (B, W, 1536) → (B, W, 512)          │
    └────────────────┬───────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────┐
    │  + Positional Embedding (learnable)    │
    │  + Subject Embedding (learnable)       │
    │  (B, W, 512) → (B, W, 512)           │
    └────────────────┬───────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────┐
    │  Transformer Encoder                   │
    │  8 capas × 8 heads                     │
    │  (B, W, 512) → (B, W, 512)           │
    └────────────────┬───────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────┐
    │  Subject-Conditional Linear            │
    │  (B, W, 512) → (B, W, num_vertices)  │
    └────────────────┬───────────────────────┘
                     │
                     ▼
    Predicción BOLD (B, W, num_vertices)
"""
import torch
import torch.nn as nn


class TemporalTransformerEncoder(nn.Module):
    """
    Transformer Encoder temporal que procesa ventanas de TRs.

    Permite intercambio de información entre timesteps dentro de una
    ventana, capturando dependencias temporales que el HRF fijo no puede.

    Args:
        d_model: Dimensión del modelo (debe coincidir con bottleneck_size).
        nhead: Número de cabezas de atención.
        num_layers: Número de capas del Transformer Encoder.
        max_window: Tamaño máximo de ventana en TRs.
        dropout: Dropout en las capas del Transformer.
        num_subjects: Número de sujetos (para subject embeddings).
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 8,
        max_window: int = 128,
        dropout: float = 0.1,
        num_subjects: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_window = max_window

        # --- Positional Embedding (learnable, como TriBE) ---
        # Cada posición en la ventana temporal tiene su propio vector aprendible.
        # A diferencia del positional encoding sinusoidal, esto permite al modelo
        # aprender representaciones arbitrarias de la posición temporal.
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_window, d_model) * 0.02
        )

        # --- Subject Embedding (opcional, learnable) ---
        # Un vector global por sujeto que se suma a todos los timesteps.
        # Permite al Transformer adaptar su procesamiento según el sujeto.
        if num_subjects > 0:
            self.subject_embedding = nn.Embedding(num_subjects, d_model)
        else:
            self.subject_embedding = None

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # FFN estándar: 4x expansion
            dropout=dropout,
            batch_first=True,             # (B, SeqLen, d_model) format
            norm_first=True,              # Pre-LayerNorm (más estable)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # --- LayerNorm final ---
        self.final_norm = nn.LayerNorm(d_model)

        # Inicialización Xavier para los parámetros del Transformer
        self._init_weights()

    def _init_weights(self):
        """Inicialización de pesos al estilo del paper original de Transformers."""
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        subject_idx: int = None,
    ) -> torch.Tensor:
        """
        Forward pass del Transformer temporal.

        Args:
            x: (B, W, d_model) — features comprimidos de una ventana de TRs.
            subject_idx: Índice numérico del sujeto (para subject embedding).

        Returns:
            (B, W, d_model) — features enriquecidos con contexto temporal.
        """
        B, W, D = x.shape
        assert W <= self.max_window, (
            f"Ventana ({W}) excede max_window ({self.max_window})"
        )

        # 1. Sumar positional embedding (recortado a la ventana actual)
        x = x + self.pos_embedding[:, :W, :]

        # 2. Sumar subject embedding (si aplica)
        if self.subject_embedding is not None and subject_idx is not None:
            subj_emb = self.subject_embedding(
                torch.tensor(subject_idx, device=x.device)
            )  # (d_model,)
            x = x + subj_emb.unsqueeze(0).unsqueeze(0)  # broadcast: (1, 1, d_model)

        # 3. Transformer Encoder
        x = self.encoder(x)

        # 4. LayerNorm final
        x = self.final_norm(x)

        return x
