"""
inspect_gemma4_modules.py — Inspecciona la arquitectura interna de Gemma 4 E2B-it.

Carga el modelo en CPU (sin forward pass) e imprime todos los módulos
relacionados con vision, audio y los projectors. Necesario para conocer
los nombres exactos antes de registrar los hooks en extract_features_v3.py.

Uso:
    python -m src.inspect_gemma4_modules

Salida esperada:
    vision_model                  SiglipVisionModel
    vision_model.encoder.layers.N SiglipEncoderLayer    ← última capa ViT
    multi_modal_projector         Gemma4MultiModalProjector ← corte visual
    audio_model                   Gemma4AudioEncoder
    audio_model.conformer_layers.N Gemma4ConformerLayer  ← última capa Conf.
    audio_multi_modal_projector   Linear                 ← corte audio
"""

import sys
from transformers import AutoModelForMultimodalLM, AutoConfig

MODEL_ID = "google/gemma-4-E2B-it"

# Palabras clave que nos interesan para los hooks
KEYWORDS = [
    "vision", "vit", "image", "siglip",
    "audio", "conformer", "speech", "usm",
    "project", "multi_modal",
]


def main():
    print("=" * 70)
    print(f"Inspección de módulos: {MODEL_ID}")
    print("Cargando en CPU (solo metadatos, sin forward pass)...")
    print("=" * 70)

    # Carga solo el config primero (instantáneo)
    config = AutoConfig.from_pretrained(MODEL_ID)
    print(f"\n[CONFIG] Tipo de modelo: {config.model_type}")
    if hasattr(config, "vision_config"):
        vc = config.vision_config
        print(f"[CONFIG] vision_config.hidden_size:     {getattr(vc, 'hidden_size', 'N/A')}")
        print(f"[CONFIG] vision_config.num_hidden_layers: {getattr(vc, 'num_hidden_layers', 'N/A')}")
    if hasattr(config, "audio_config"):
        ac = config.audio_config
        print(f"[CONFIG] audio_config.hidden_size:      {getattr(ac, 'hidden_size', 'N/A')}")
        print(f"[CONFIG] audio_config.num_hidden_layers: {getattr(ac, 'num_hidden_layers', 'N/A')}")
    if hasattr(config, "text_config"):
        tc = config.text_config
        print(f"[CONFIG] text_config.hidden_size:       {getattr(tc, 'hidden_size', 'N/A')}")
        print(f"[CONFIG] text_config.num_hidden_layers:  {getattr(tc, 'num_hidden_layers', 'N/A')}")

    print("\n" + "-" * 70)
    print("Cargando pesos (CPU, bfloat16 si es posible)...")
    print("-" * 70)

    try:
        import torch
        model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        print(f"\n[ERROR] No se pudo cargar el modelo: {e}")
        print("Revisa tu conexión a HuggingFace o que tengas acceso al modelo.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("MÓDULOS RELEVANTES (vision, audio, projectors):")
    print("=" * 70)

    found = []
    for name, module in model.named_modules():
        lower = name.lower()
        if any(k in lower for k in KEYWORDS):
            class_name = type(module).__name__
            found.append((name, class_name))

    if not found:
        print("[ADVERTENCIA] No se encontraron módulos con las keywords esperadas.")
        print("Imprimiendo TODOS los módulos (primeros 60):")
        for i, (name, module) in enumerate(model.named_modules()):
            if i >= 60:
                break
            print(f"  {name:60s} {type(module).__name__}")
    else:
        for name, class_name in found:
            # Destacar los projectors (donde cortamos)
            marker = "  ◄── CORTE AQUÍ" if "project" in name.lower() else ""
            print(f"  {name:60s} {class_name}{marker}")

    # Identificar automáticamente el índice de la última capa
    print("\n" + "=" * 70)
    print("RESUMEN DE CAPAS FINALES (para los hooks):")
    print("=" * 70)

    # ViT
    vit_layers = [(n, m) for n, m in model.named_modules()
                  if "vision" in n.lower() and "encoder.layers" in n.lower()
                  and n.count(".") == 3]  # depth exacta de la capa
    if vit_layers:
        last_vit = vit_layers[-1]
        print(f"\n[ViT] Última capa encoder:  {last_vit[0]}")
        print(f"      Clase:                 {type(last_vit[1]).__name__}")

    # Conformer
    conf_layers = [(n, m) for n, m in model.named_modules()
                   if "audio" in n.lower() and ("conformer" in n.lower() or "encoder" in n.lower())
                   and "layers" in n.lower()
                   and n.count(".") <= 4]
    if conf_layers:
        last_conf = conf_layers[-1]
        print(f"\n[Conformer] Última capa:   {last_conf[0]}")
        print(f"            Clase:          {type(last_conf[1]).__name__}")

    # Projectors
    projectors = [(n, m) for n, m in model.named_modules()
                  if "project" in n.lower() and "." not in n.replace("multi_modal", "")]
    if projectors:
        print("\n[Projectors encontrados]:")
        for n, m in projectors:
            print(f"  {n:50s} {type(m).__name__}")

    # Output dimensions del primer forward con tensores dummy
    print("\n" + "=" * 70)
    print("DIMENSIONES DE SALIDA (forward parcial en CPU):")
    print("=" * 70)

    import torch
    try:
        # Test del ViT con una imagen dummy pequeña
        if hasattr(model, "vision_model"):
            dummy_pixel = torch.zeros(1, 3, 336, 336, dtype=torch.bfloat16)
            with torch.no_grad():
                vit_out = model.vision_model(dummy_pixel)
            if hasattr(vit_out, "last_hidden_state"):
                print(f"\n[ViT] last_hidden_state shape: {vit_out.last_hidden_state.shape}")
                print(f"      → Dim por patch: {vit_out.last_hidden_state.shape[-1]}")
                print(f"      → Num patches:   {vit_out.last_hidden_state.shape[1]}")
    except Exception as e:
        print(f"\n[ViT forward] Falló: {e}")
        print("  → Necesitarás confirmar las dimensiones manualmente del config.")

    try:
        if hasattr(model, "audio_model"):
            # Test del Conformer con audio dummy (1 segundo a 16kHz → 160 muestras)
            # El Conformer espera features de audio preprocesados, no raw waveform
            print("\n[Conformer] Audio_model existe. Dimensión de salida a confirmar en extracción.")
            if hasattr(config, "audio_config"):
                print(f"  → audio_config.hidden_size: {getattr(config.audio_config, 'hidden_size', 'N/A')}")
    except Exception as e:
        print(f"\n[Conformer forward] Falló: {e}")

    print("\n" + "=" * 70)
    print("Guarda esta salida para completar extract_features_v3.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
