
python -m llmcal sst2_256_812 basic_sst2_0-shot_litgpt lm_llama3 lora_500samples no_calibration --accelerator "gpu"

python -m llmcal agnews_4_2951 basic_agnews_0-shot_litgpt lm_llama3 no_adaptation_bf16 affine_scalar \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30
python -m llmcal agnews_4_2951 basic_agnews_0-shot_litgpt lm_llama3 no_adaptation_bf16 temp_scaling \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30