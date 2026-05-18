"""
Validador de sincronización entre tensores de estímulo (Gemma) y fMRI (BOLD).

Ejecuta un diagnóstico completo de salud de los tensores antes de entrenar,
verificando que no haya NaN, Inf, filas vacías, ni desajustes dimensionales.

Uso:
    uv run python src/validate_tensors.py
    uv run python src/validate_tensors.py --features data/features_v2/real_stimulus_features.pt --bold data/features_v2/fmri/sub-01.pt
"""

import argparse
import torch

def validate_tensor(path: str, name: str) -> dict:
    """Valida un tensor individual y retorna un reporte."""
    print(f"\n{'─'*50}")
    print(f"📊 Validando: {name}")
    print(f"   Path: {path}")
    print(f"{'─'*50}")
    
    t = torch.load(path, weights_only=True)
    
    report = {
        "name": name,
        "shape": t.shape,
        "dtype": t.dtype,
        "nan_count": t.isnan().sum().item(),
        "inf_count": t.isinf().sum().item(),
        "zero_rows": (t.abs().sum(dim=1) == 0).sum().item(),
        "min": t.min().item(),
        "max": t.max().item(),
        "mean": t.mean().item(),
        "std": t.std().item(),
        "total_elements": t.numel(),
    }
    
    print(f"  Forma:        {report['shape']}")
    print(f"  Dtype:        {report['dtype']}")
    print(f"  Rango:        [{report['min']:.6f}, {report['max']:.6f}]")
    print(f"  Media ± Std:  {report['mean']:.6f} ± {report['std']:.6f}")
    print(f"  NaN:          {report['nan_count']} / {report['total_elements']}")
    print(f"  Inf:          {report['inf_count']} / {report['total_elements']}")
    print(f"  Filas cero:   {report['zero_rows']} / {report['shape'][0]}")
    
    # Diagnóstico por fila (primeras y últimas 3)
    print("\n  Primeras 3 filas:")
    for i in range(min(3, t.shape[0])):
        row = t[i]
        print(f"    [{i:>3d}] mean={row.mean().item():>10.6f}  std={row.std().item():>10.6f}  nan={row.isnan().sum().item()}")
    
    print("  Últimas 3 filas:")
    for i in range(max(0, t.shape[0]-3), t.shape[0]):
        row = t[i]
        print(f"    [{i:>3d}] mean={row.mean().item():>10.6f}  std={row.std().item():>10.6f}  nan={row.isnan().sum().item()}")
    
    # Veredicto
    healthy = True
    issues = []
    
    if report['nan_count'] > 0:
        issues.append(f"❌ Contiene {report['nan_count']} valores NaN")
        healthy = False
    if report['inf_count'] > 0:
        issues.append(f"❌ Contiene {report['inf_count']} valores Inf")
        healthy = False
    if report['zero_rows'] > 0:
        issues.append(f"⚠️  {report['zero_rows']} filas son todo-cero (posibles fallbacks de extracción)")
        if report['zero_rows'] > report['shape'][0] * 0.1:  # >10% es peligroso
            healthy = False
    if report['std'] < 1e-6:
        issues.append(f"❌ Std demasiado baja ({report['std']:.8f}), tensor posiblemente constante")
        healthy = False
    
    if healthy and not issues:
        print("\n  ✅ SALUDABLE — Tensor listo para entrenamiento.")
    elif healthy and issues:
        for issue in issues:
            print(f"  {issue}")
        print("\n  ⚠️  ADVERTENCIA — Puede funcionar pero revisa los warnings.")
    else:
        for issue in issues:
            print(f"  {issue}")
        print("\n  ❌ NO SALUDABLE — Regenera este tensor antes de entrenar.")
    
    report['healthy'] = healthy
    report['issues'] = issues
    return report


def validate_synchronization(
    features_path: str,
    bold_path: str,
    hrf_delay: float = 5.0,
    fmri_tr: float = 1.49,
):
    """Valida que los tensores estén sincronizados y listos para entrenar."""
    
    print("\n" + "=" * 60)
    print("🔬 VALIDACIÓN COMPLETA DE TENSORES")
    print("=" * 60)
    
    # 1. Validar cada tensor individual
    feat_report = validate_tensor(features_path, "Estímulo (Gemma 4)")
    bold_report = validate_tensor(bold_path, "Cerebro (fMRI BOLD)")
    
    # 2. Validar sincronización temporal
    print(f"\n{'─'*50}")
    print("🔗 Validando SINCRONIZACIÓN")
    print(f"{'─'*50}")
    
    delay_in_trs = int(hrf_delay / fmri_tr)
    t_stim = feat_report['shape'][0]
    t_fmri = bold_report['shape'][0]
    
    # Simular la alineación HRF
    valid_fmri = t_fmri - delay_in_trs
    aligned_samples = min(valid_fmri, t_stim)
    
    print(f"  TRs de estímulo (Gemma):    {t_stim}")
    print(f"  TRs de fMRI (BOLD):         {t_fmri}")
    print(f"  HRF delay:                  {hrf_delay}s = {delay_in_trs} TRs")
    print(f"  TR del escáner:             {fmri_tr}s")
    print(f"  fMRI válidos (post-HRF):    {valid_fmri}")
    print(f"  Muestras alineadas finales: {aligned_samples}")
    
    if t_stim == 0 or t_fmri == 0:
        print("\n  ❌ ERROR: Uno o ambos tensores están vacíos.")
        return False
    
    if aligned_samples < 100:
        print(f"\n  ⚠️  ADVERTENCIA: Solo {aligned_samples} muestras — puede ser insuficiente.")
    
    # Dimensiones de features
    feat_dim = feat_report['shape'][1]
    bold_dim = bold_report['shape'][1]
    print(f"\n  Dim features (Gemma hidden): {feat_dim}")
    print(f"  Dim BOLD (vóxeles):          {bold_dim}")
    
    if feat_dim != 1536:
        print(f"  ⚠️  Se esperaba feat_dim=1536, encontrado {feat_dim}")
    
    # 3. Reporte final
    all_healthy = feat_report['healthy'] and bold_report['healthy']
    
    print("\n" + "="*60)
    if all_healthy and aligned_samples >= 100:
        print("✅ TODO LISTO PARA ENTRENAR")
        print(f"   Muestras disponibles: {aligned_samples}")
        print(f"   Arquitectura: Linear(1536→512→{bold_dim})")
        print("\n   Comando:")
        print(f"   uv run python train.py --subject sub-01 --epochs 50 --fmri_tr {fmri_tr}")
    else:
        print("❌ HAY PROBLEMAS — Revisa los errores anteriores.")
        if not feat_report['healthy']:
            print("   → Regenera features: uv run python -m src.extract_features ...")
        if not bold_report['healthy']:
            print("   → Regenera fMRI: uv run python src/prepare_fmri.py ...")
    print(f"{'='*60}")
    
    return all_healthy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validador de tensores pre-entrenamiento")
    parser.add_argument("--features", type=str, default="data/features_v2/real_stimulus_features.pt")
    parser.add_argument("--bold", type=str, default="data/features_v2/fmri/sub-01.pt")
    parser.add_argument("--hrf_delay", type=float, default=5.0)
    parser.add_argument("--fmri_tr", type=float, default=1.49, help="Tiempo de repetición (TR) del escáner en segs")
    
    args = parser.parse_args()
    
    validate_synchronization(
        features_path=args.features,
        bold_path=args.bold,
        hrf_delay=args.hrf_delay,
        fmri_tr=args.fmri_tr,
    )
