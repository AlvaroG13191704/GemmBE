"""
Ridge Regression Model — Baseline lineal con sklearn.

No usa gradientes de PyTorch. En on_train_start carga TODO el dataset
de entrenamiento y ajusta RidgeCV. Luego forward usa los coeficientes.
"""

import torch
import torch.nn as nn
# pyrefly: ignore [missing-import]
from src.models.base_module import BrainEncodingModule


class RidgeModel(BrainEncodingModule):
    """
    Ridge Regression como baseline lineal.
    
    Entrena RidgeCV con validación cruzada de alpha en on_train_start,
    usando todo el dataset de entrenamiento (no mini-batches).
    """

    def __init__(self, **kwargs):
        # Quitar lr y weight_decay de kwargs para Ridge (no los usa)
        kwargs.setdefault("lr", 0.0)
        kwargs.setdefault("weight_decay", 0.0)
        super().__init__(model_name="ridge", **kwargs)
        
        # Dummy parameter para que Lightning no se queje de "no parameters"
        self.dummy = nn.Parameter(torch.zeros(1))
        self.ridge = None

    def on_train_start(self):
        """Entrena Ridge con TODO el train set (una sola vez)."""
        if self.ridge is not None:
            return
        
        print("\n📐 Entrenando RidgeCV con todo el dataset...")
        X_list, y_list = [], []
        train_loader = self.trainer.train_dataloader
        for batch in train_loader:
            X_list.append(batch[0])
            y_list.append(batch[1])
        
        X = torch.cat(X_list).cpu().numpy()
        y = torch.cat(y_list).cpu().numpy()
        
        from sklearn.linear_model import RidgeCV
        self.ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0], cv=3)
        self.ridge.fit(X, y)
        print(f"   Mejor alpha: {self.ridge.alpha_}")
        
        # Evaluar en train para logging
        pred = torch.from_numpy(self.ridge.predict(X)).float()
        y_t = torch.from_numpy(y).float()
        loss = nn.functional.mse_loss(pred, y_t)
        pearson = self._compute_pearson(pred, y_t)
        print(f"   Train MSE: {loss:.6f} | Pearson: {pearson.mean():.4f}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.ridge is None:
            return torch.zeros(features.size(0), self.hparams.num_vertices, device=features.device)
        X = features.cpu().numpy()
        pred = self.ridge.predict(X)
        return torch.from_numpy(pred).to(features.device).float()

    def training_step(self, batch, batch_idx):
        # Ridge ya está entrenado; solo loggear un dummy loss
        if batch_idx == 0 and self.ridge is not None:
            features, bold = batch
            pred = self(features)
            loss = nn.functional.mse_loss(pred, bold)
            self.log("train/loss", loss, on_epoch=True)
        return self.dummy * 0.0  # dummy loss con grad

    def configure_optimizers(self):
        # Dummy optimizer (no se usa realmente)
        return torch.optim.SGD([self.dummy], lr=1e-3)
