"""
MetricsCallback — Guarda Pearson maps y resultados de test para plotting posterior.
"""

import json
from pathlib import Path

import torch
import lightning as L


class MetricsCallback(L.Callback):
    """
    Al final de cada validación y test, guarda:
      - pearson_map.pt: correlación por parcela (1000,)
      - test_results.json: métricas agregadas
    """

    def __init__(self, save_dir: str):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if hasattr(pl_module, "_last_val_pearson_map"):
            pearson = pl_module._last_val_pearson_map
            out_path = self.save_dir / "pearson_map_val.pt"
            torch.save(pearson, out_path)

    def on_test_epoch_end(self, trainer, pl_module):
        results = {}
        
        if hasattr(pl_module, "_last_test_pearson_map"):
            pearson = pl_module._last_test_pearson_map
            out_path = self.save_dir / "pearson_map_test.pt"
            torch.save(pearson, out_path)
            results["test_pearson_mean"] = pearson.mean().item()
            results["test_pearson_std"] = pearson.std().item()
            results["test_pearson>0.15"] = (pearson > 0.15).sum().item()

        # Recolectar métricas loggeadas
        metrics = trainer.callback_metrics
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                results[k.replace("/", "_")] = v.item()

        # Guardar JSON
        with open(self.save_dir / "test_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Métricas guardadas en: {self.save_dir}")
