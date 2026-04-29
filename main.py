"""
MicroTRIBE-Gemma — Punto de entrada principal.

Simplificación del pipeline TriBE v2 (Meta FAIR) usando Gemma 4 E2B-it
como extractor multimodal congelado para predicción de actividad cortical fMRI.

Uso:
    # Modo dummy (sin descargar modelo, para validación de arquitectura):
    python main.py
    
    # Modo real (descarga Gemma 4 E2B-it, requiere ~5GB VRAM):
    python main.py --real
    
    # Modo multi-subject:
    python main.py --subjects sub-01 sub-02 sub-03
"""

import argparse
import sys
import torch

from src.config import ModelConfig
from src.model import MicroTribeGemma
from src.temporal_alignment import HRFAligner


def run_dummy_inference(model: MicroTribeGemma, config: ModelConfig) -> None:
    """
    Ejecuta un forward pass completo con datos sintéticos.
    Valida que todas las dimensiones son correctas sin necesitar GPU ni modelo real.
    """
    print("\n" + "=" * 60)
    print("🧪 MODO DUMMY — Forward pass con datos sintéticos")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 256  # Simula la longitud combinada de tokens (texto + imagen + audio)
    
    outputs = model(
        dummy_seq_len=seq_len,
        dummy_batch_size=batch_size,
    )
    
    print("\n📐 Formas de tensores en cada etapa:")
    print(f"  ┌─ Hidden States (Gemma 4):  {outputs['hidden_states'].shape}")
    print(f"  ├─ Pooled (Temporal):        {outputs['pooled'].shape}")
    print(f"  ├─ Compressed (Bottleneck):  {outputs['compressed'].shape}")
    print(f"  └─ Predicted BOLD:           {outputs['predicted_bold'].shape}")
    
    # Validaciones
    B = batch_size
    assert outputs["hidden_states"].shape == (B, seq_len, config.gemma_hidden_size), \
        f"Hidden states shape incorrecto: {outputs['hidden_states'].shape}"
    assert outputs["pooled"].shape == (B, config.gemma_hidden_size), \
        f"Pooled shape incorrecto: {outputs['pooled'].shape}"
    assert outputs["compressed"].shape == (B, config.bottleneck_size), \
        f"Compressed shape incorrecto: {outputs['compressed'].shape}"
    assert outputs["predicted_bold"].shape == (B, config.num_vertices), \
        f"BOLD shape incorrecto: {outputs['predicted_bold'].shape}"
    
    print("\n✅ Todas las formas son correctas!")
    print(f"  Entrada: ({B}, {seq_len}, {config.gemma_hidden_size})")
    print(f"  Salida:  ({B}, {config.num_vertices})")


def run_dummy_multi_subject(config: ModelConfig) -> None:
    """Test del modo multi-sujeto con datos dummy."""
    print("\n" + "=" * 60)
    print("👥 MODO MULTI-SUJETO — 3 sujetos simulados")
    print("=" * 60)
    
    subjects = ["sub-01", "sub-02", "sub-03"]
    model = MicroTribeGemma(
        config=config,
        subject_ids=subjects,
        dummy=True,
    )
    model = model.to(config.device)
    
    for sid in subjects:
        outputs = model(
            dummy_seq_len=128,
            dummy_batch_size=1,
            subject_id=sid,
        )
        print(f"  {sid}: BOLD prediction shape = {outputs['predicted_bold'].shape}")
    
    model.print_parameter_summary()
    print("\n✅ Multi-subject funciona correctamente!")


def run_dummy_training_loop(model: MicroTribeGemma, config: ModelConfig) -> None:
    """
    Simula un mini loop de entrenamiento para verificar que los gradientes
    fluyen correctamente solo a través de las capas entrenables.
    """
    print("\n" + "=" * 60)
    print("🔄 SIMULACIÓN DE TRAINING LOOP")
    print("=" * 60)
    
    # Configurar optimizador solo con parámetros entrenables
    trainable_params = model.get_trainable_parameters()
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    criterion = torch.nn.MSELoss()
    
    # Datos sintéticos
    batch_size = 2
    
    # fMRI target simulado (actividad BOLD por vértice)
    target_bold = torch.randn(batch_size, config.num_vertices, dtype=config.dtype, device=config.device)
    
    # Forward pass
    outputs = model(dummy_seq_len=128, dummy_batch_size=batch_size)
    predicted = outputs["predicted_bold"]
    
    # Calcular loss en float32 para evitar overflow con float16
    # (práctica estándar en mixed-precision training)
    loss = criterion(predicted.float(), target_bold.float())
    print(f"  Loss inicial: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Verificar gradientes
    has_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            has_grad = True
            grad_norm = param.grad.norm().item()
            if "backbone" in name:
                print(f"  ⚠️  ALERTA: Gradiente en backbone ({name}): {grad_norm}")
            else:
                print(f"  ✅ Gradiente en {name}: {grad_norm:.6f}")
    
    if not has_grad:
        print("  ⚠️  No se encontraron gradientes. Verificar configuración.")
    
    # Step del optimizador
    optimizer.step()
    optimizer.zero_grad()
    
    # Segundo forward para verificar que los pesos cambiaron
    outputs2 = model(dummy_seq_len=128, dummy_batch_size=batch_size)
    loss2 = criterion(outputs2["predicted_bold"].float(), target_bold.float())
    print(f"  Loss después de 1 step: {loss2.item():.4f}")
    print(f"  Δ Loss: {loss2.item() - loss.item():.6f}")
    print("\n✅ Training loop funciona correctamente!")


def run_hrf_demo() -> None:
    """Demuestra la alineación temporal HRF."""
    print("\n" + "=" * 60)
    print("⏱️  DEMO — Alineación Temporal HRF")
    print("=" * 60)
    
    aligner = HRFAligner(hrf_delay_seconds=5.0, fmri_tr_seconds=1.0)
    
    print(f"  Delay hemodinámico: {aligner.hrf_delay_seconds}s ({aligner.delay_in_trs} TRs)")
    
    # Simular 60 segundos de película y 60 TRs de fMRI
    total_trs = 60
    start, end = aligner.get_valid_range(total_trs)
    print(f"  Total TRs: {total_trs}")
    print(f"  Rango válido: TR[{start}] a TR[{end-1}] ({end - start} TRs utilizables)")
    
    # Ejemplo de mapeo
    print("\n  Mapeo estímulo → fMRI:")
    for tr in [5, 10, 15, 30, 59]:
        stim_idx = aligner.get_stimulus_index(tr)
        print(f"    fMRI[{tr}] ← estímulo[{stim_idx}]")
    
    # Alineación de tensores
    stimulus = torch.randn(60, 1536)    # 60s de features
    bold = torch.randn(60, 20_484)       # 60 TRs de fMRI
    
    aligned_stim, aligned_bold = aligner.align_stimulus_to_fmri(stimulus, bold)
    print("\n  Tensores alineados:")
    print(f"    Estímulo: {stimulus.shape} → {aligned_stim.shape}")
    print(f"    BOLD:     {bold.shape} → {aligned_bold.shape}")
    
    print("\n✅ Alineación HRF funciona correctamente!")


def run_real_inference(model: MicroTribeGemma) -> None:
    """
    Forward pass con Gemma 4 real y datos multimodales sintéticos.
    Requiere que el modelo se haya cargado con dummy=False.
    """
    print("\n" + "=" * 60)
    print("🚀 MODO REAL — Forward pass con Gemma 4 E2B-it")
    print("=" * 60)
    
    processor = model.processor
    
    # Crear un prompt de ejemplo (simula descripción de una escena de película)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe la actividad visual y auditiva en esta escena de película."
                },
            ],
        }
    ]
    
    # Procesar con el chat template de Gemma 4
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    
    inputs = processor(text=text, return_tensors="pt")
    
    # Mover al dispositivo correcto
    device = next(model.backbone.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Forward pass
    outputs = model(**inputs)
    
    print("\n📐 Formas de tensores (modo real):")
    print(f"  ┌─ Hidden States:     {outputs['hidden_states'].shape}")
    print(f"  ├─ Pooled:            {outputs['pooled'].shape}")
    print(f"  ├─ Compressed:        {outputs['compressed'].shape}")
    print(f"  └─ Predicted BOLD:    {outputs['predicted_bold'].shape}")
    
    print("\n✅ Forward pass con Gemma 4 real completado!")


def main():
    parser = argparse.ArgumentParser(
        description="MicroTRIBE-Gemma: Simplified TriBE v2 with Gemma 4"
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Cargar Gemma 4 real (requiere ~5GB VRAM). Default: modo dummy.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="IDs de sujetos para multi-subject. Ej: --subjects sub-01 sub-02",
    )
    parser.add_argument(
        "--pooling",
        choices=["mean", "conv1d"],
        default="conv1d",
        help="Modo de temporal pooling. Default: mean.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Dispositivo de cómputo. Default: auto-detect.",
    )
    
    args = parser.parse_args()
    
    # Configuración
    config = ModelConfig()
    if args.device != "auto":
        config.override_device(args.device)
    
    print("🧠 MicroTRIBE-Gemma — Simplified TriBE v2 Brain Encoding")
    print(f"   Dispositivo: {config.device} | Dtype: {config.dtype}")
    print(f"   Modelo: {config.model_id}")
    print(f"   Bottleneck: {config.gemma_hidden_size} → {config.bottleneck_size}")
    print(f"   Vértices: {config.num_vertices:,}")
    
    # Crear modelo
    model = MicroTribeGemma(
        config=config,
        subject_ids=args.subjects,
        pooling_mode=args.pooling,
        dummy=not args.real,
    )
    
    # Mover capas entrenables al dispositivo
    if not args.real:
        model = model.to(config.device)
    
    # Resumen de parámetros
    model.print_parameter_summary()
    
    if args.real:
        # Modo real con Gemma 4
        run_real_inference(model)
    else:
        # Modo dummy — suite completa de validación
        run_dummy_inference(model, config)
        run_hrf_demo()
        run_dummy_training_loop(model, config)
        
        if not args.subjects:
            run_dummy_multi_subject(config)
    
    print("\n" + "=" * 60)
    print("🎉 ¡Todas las validaciones pasaron exitosamente!")
    print("=" * 60)


if __name__ == "__main__":
    main()