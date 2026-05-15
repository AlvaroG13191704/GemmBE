# Fase 1: Extraer piloto multimodal
uv run python -m src.extract_features --pilot --sub_samples 1 --max_context_images 3

# Fase 2: Extraer piloto text-only (mismos chunks, solo transcript)
uv run python -m src.extract_features --pilot --text_only --sub_samples 1 --max_context_images 3

# Fase 3: Preparar fMRI filtrado
uv run python src/filter_fmri.py --all_subjects

# Fase 4: Entrenar baselines y modelos
uv run python src/baselines.py              # Ridge Regression
uv run python src/ablations.py              # Sin Bottleneck, Sin HRF
uv run python train.py --all_subjects       # Tu modelo completo

# Fase 5: Generar visualizaciones
# (script que preparemos con matplotlib/nilearn para los brain maps)


# 1. Extraer estímulos multimodales (piloto)
uv run python -m src.extract_features --pilot --sub_samples 1 --max_context_images 3

# 2. Extraer estímulos text-only (baseline)
uv run python -m src.extract_features --pilot --text_only --sub_samples 1 --max_context_images 3

# 3. Fusionar y filtrar fMRI
uv run python -m src.extract_features --merge
uv run python src/filter_fmri.py --all_subjects

# 4. Validar alineación
uv run python src/validate_tensors.py \
    --features data/features/real_stimulus_features.pt \
    --bold data/subjects_fmri_filtered/sub-01.pt

# 5. Entrenar los 4 sujetos individualmente
for s in sub-01 sub-02 sub-03 sub-05; do
    uv run python train.py --subject $s --mode full --epochs 100 \
        --fmri_dir ./data/subjects_fmri_filtered
done


# 1. En RunPod/Lambda con GPU L4 (~$0.50/hora):
git clone <tu-repo>
cd GemmaBe
bash cloud/setup.sh

# 2. Lanzar extracción (~4-6 horas para 24 chunks)
bash cloud/run_extraction.sh

# 3. Descargar resultados
tar czf results_v2.tar.gz data/features_v2/ data/features_v2_text_only/
