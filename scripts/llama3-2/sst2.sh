# #!/bin/bash

### No adaptation + no calibration
python -m llmcal sst2_16_564 basic_sst2_0-shot_litgpt llama3 no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

mkdir -p experiments/sst2_256_821/basic_sst2_0-shot_litgpt/llama3/no_adaptation_bf16/.cache/predictions
ln -sf ../../../../../../sst2_16_564/basic_sst2_0-shot_litgpt/llama3/no_adaptation_bf16/.cache/predictions/test experiments/sst2_256_821/basic_sst2_0-shot_litgpt/llama3/no_adaptation_bf16/.cache/predictions
python -m llmcal sst2_256_821 basic_sst2_0-shot_litgpt llama3 no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

mkdir -p experiments/sst2_8_932/basic_sst2_0-shot_litgpt/llama3/no_adaptation_bf16/.cache/predictions
ln -sf ../../../../../../sst2_16_564/basic_sst2_0-shot_litgpt/llama3/no_adaptation_bf16/.cache/predictions/test experiments/sst2_8_932/basic_sst2_0-shot_litgpt/llama3/no_adaptation_bf16/.cache/predictions
python -m llmcal sst2_8_932 basic_sst2_0-shot_litgpt llama3 no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

### No adaptation + affine scalar
python -m llmcal sst2_8_932 basic_sst2_0-shot_litgpt llama3 no_adaptation_bf16 affine_scalar \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

python -m llmcal sst2_16_564 basic_sst2_0-shot_litgpt llama3 no_adaptation_bf16 affine_scalar \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

python -m llmcal sst2_256_821 basic_sst2_0-shot_litgpt llama3 no_adaptation_bf16 affine_scalar \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

### No adaptation + temp scaling
python -m llmcal sst2_8_932 basic_sst2_0-shot_litgpt llama3 no_adaptation_bf16 temp_scaling \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

python -m llmcal sst2_16_564 basic_sst2_0-shot_litgpt llama3 no_adaptation_bf16 temp_scaling \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

python -m llmcal sst2_256_821 basic_sst2_0-shot_litgpt llama3 no_adaptation_bf16 temp_scaling \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

### No adaptation + bias only
python -m llmcal sst2_8_932 basic_sst2_0-shot_litgpt llama3 no_adaptation_bf16 bias_only \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

python -m llmcal sst2_16_564 basic_sst2_0-shot_litgpt llama3 no_adaptation_bf16 bias_only \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

python -m llmcal sst2_256_821 basic_sst2_0-shot_litgpt llama3 no_adaptation_bf16 bias_only \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

### Lora + no calibration
python -m llmcal sst2_8_932 basic_sst2_0-shot_litgpt llama3 lora_10samples no_calibration --accelerator "gpu"
python -m llmcal sst2_16_564 basic_sst2_0-shot_litgpt llama3 lora_20samples no_calibration --accelerator "gpu"
python -m llmcal sst2_256_821 basic_sst2_0-shot_litgpt llama3 lora_500samples no_calibration --accelerator "gpu"