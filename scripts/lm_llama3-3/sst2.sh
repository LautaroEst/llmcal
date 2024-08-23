# #!/bin/bash

### No adaptation + no calibration
python -m llmcal sst2_16_783 basic_sst2_0-shot_litgpt lm_llama3 no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

# mkdir -p experiments/sst2_256_812/basic_sst2_0-shot_litgpt/lm_llama3/no_adaptation_bf16/.cache/predictions
# ln -sf ../../../../../../sst2_16_783/basic_sst2_0-shot_litgpt/lm_llama3/no_adaptation_bf16/.cache/predictions/test experiments/sst2_256_812/basic_sst2_0-shot_litgpt/lm_llama3/no_adaptation_bf16/.cache/predictions
# python -m llmcal sst2_256_812 basic_sst2_0-shot_litgpt lm_llama3 no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

mkdir -p experiments/sst2_512_767/basic_sst2_0-shot_litgpt/lm_llama3/no_adaptation_bf16/.cache/predictions
ln -sf ../../../../../../sst2_16_783/basic_sst2_0-shot_litgpt/lm_llama3/no_adaptation_bf16/.cache/predictions/test experiments/sst2_512_767/basic_sst2_0-shot_litgpt/lm_llama3/no_adaptation_bf16/.cache/predictions
python -m llmcal sst2_512_767 basic_sst2_0-shot_litgpt lm_llama3 no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

mkdir -p experiments/sst2_8_923/basic_sst2_0-shot_litgpt/lm_llama3/no_adaptation_bf16/.cache/predictions
ln -sf ../../../../../../sst2_16_783/basic_sst2_0-shot_litgpt/lm_llama3/no_adaptation_bf16/.cache/predictions/test experiments/sst2_8_923/basic_sst2_0-shot_litgpt/lm_llama3/no_adaptation_bf16/.cache/predictions
python -m llmcal sst2_8_923 basic_sst2_0-shot_litgpt lm_llama3 no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

# mkdir -p experiments/sst2_32_1783/basic_sst2_0-shot_litgpt/lm_llama3/no_adaptation_bf16/.cache/predictions
# ln -sf ../../../../../../sst2_16_783/basic_sst2_0-shot_litgpt/lm_llama3/no_adaptation_bf16/.cache/predictions/test experiments/sst2_32_1783/basic_sst2_0-shot_litgpt/lm_llama3/no_adaptation_bf16/.cache/predictions
# python -m llmcal sst2_32_1783 basic_sst2_0-shot_litgpt lm_llama3 no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1


### No adaptation + affine scalar
python -m llmcal sst2_8_923 basic_sst2_0-shot_litgpt lm_llama3 no_adaptation_bf16 affine_scalar \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

# python -m llmcal sst2_16_783 basic_sst2_0-shot_litgpt lm_llama3 no_adaptation_bf16 affine_scalar \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

# python -m llmcal sst2_32_1783 basic_sst2_0-shot_litgpt lm_llama3 no_adaptation_bf16 affine_scalar \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30


# python -m llmcal sst2_256_812 basic_sst2_0-shot_litgpt lm_llama3 no_adaptation_bf16 affine_scalar \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

python -m llmcal sst2_512_767 basic_sst2_0-shot_litgpt lm_llama3 no_adaptation_bf16 affine_scalar \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

### No adaptation + temp scaling
python -m llmcal sst2_8_923 basic_sst2_0-shot_litgpt lm_llama3 no_adaptation_bf16 temp_scaling \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

# python -m llmcal sst2_16_783 basic_sst2_0-shot_litgpt lm_llama3 no_adaptation_bf16 temp_scaling \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

# python -m llmcal sst2_32_1783 basic_sst2_0-shot_litgpt lm_llama3 no_adaptation_bf16 temp_scaling \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

# python -m llmcal sst2_256_812 basic_sst2_0-shot_litgpt lm_llama3 no_adaptation_bf16 temp_scaling \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

python -m llmcal sst2_512_767 basic_sst2_0-shot_litgpt lm_llama3 no_adaptation_bf16 temp_scaling \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

### No adaptation + bias only
python -m llmcal sst2_8_923 basic_sst2_0-shot_litgpt lm_llama3 no_adaptation_bf16 bias_only \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

# python -m llmcal sst2_16_783 basic_sst2_0-shot_litgpt lm_llama3 no_adaptation_bf16 bias_only \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

# python -m llmcal sst2_32_1783 basic_sst2_0-shot_litgpt lm_llama3 no_adaptation_bf16 bias_only \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

# python -m llmcal sst2_256_812 basic_sst2_0-shot_litgpt lm_llama3 no_adaptation_bf16 bias_only \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30


python -m llmcal sst2_512_767 basic_sst2_0-shot_litgpt lm_llama3 no_adaptation_bf16 bias_only \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

### Lora + no calibration
python -m llmcal sst2_8_923 basic_sst2_0-shot_litgpt lm_llama3 lora_10samples no_calibration --accelerator "gpu"
# python -m llmcal sst2_16_783 basic_sst2_0-shot_litgpt lm_llama3 lora_20samples no_calibration --accelerator "gpu"
# python -m llmcal sst2_32_1783 basic_sst2_0-shot_litgpt lm_llama3 lora_40samples no_calibration --accelerator "gpu"
# python -m llmcal sst2_256_812 basic_sst2_0-shot_litgpt lm_llama3 lora_500samples no_calibration --accelerator "gpu"
python -m llmcal sst2_512_767 basic_sst2_0-shot_litgpt lm_llama3 lora_1000samples no_calibration --accelerator "gpu"