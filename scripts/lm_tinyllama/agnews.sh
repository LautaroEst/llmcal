#!/bin/bash

### No adaptation + no calibration
python -m llmcal agnews_16_738 basic_agnews_0-shot_litgpt lm_tinyllama no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

mkdir -p experiments/agnews_256_493/basic_agnews_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions
ln -sf ../../../../../../agnews_16_738/basic_agnews_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions/test experiments/agnews_256_493/basic_agnews_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions
python -m llmcal agnews_256_493 basic_agnews_0-shot_litgpt lm_tinyllama no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

mkdir -p experiments/agnews_8_639/basic_agnews_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions
ln -sf ../../../../../../agnews_16_738/basic_agnews_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions/test experiments/agnews_8_639/basic_agnews_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions
python -m llmcal agnews_8_639 basic_agnews_0-shot_litgpt lm_tinyllama no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

### No adaptation + affine scalar
python -m llmcal agnews_8_639 basic_agnews_0-shot_litgpt lm_tinyllama no_adaptation_bf16 affine_scalar \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

python -m llmcal agnews_16_738 basic_agnews_0-shot_litgpt lm_tinyllama no_adaptation_bf16 affine_scalar \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

python -m llmcal agnews_256_493 basic_agnews_0-shot_litgpt lm_tinyllama no_adaptation_bf16 affine_scalar \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

### No adaptation + temp scaling
python -m llmcal agnews_8_639 basic_agnews_0-shot_litgpt lm_tinyllama no_adaptation_bf16 temp_scaling \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

python -m llmcal agnews_16_738 basic_agnews_0-shot_litgpt lm_tinyllama no_adaptation_bf16 temp_scaling \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

python -m llmcal agnews_256_493 basic_agnews_0-shot_litgpt lm_tinyllama no_adaptation_bf16 temp_scaling \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

### No adaptation + bias only
python -m llmcal agnews_8_639 basic_agnews_0-shot_litgpt lm_tinyllama no_adaptation_bf16 bias_only \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

python -m llmcal agnews_16_738 basic_agnews_0-shot_litgpt lm_tinyllama no_adaptation_bf16 bias_only \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

python -m llmcal agnews_256_493 basic_agnews_0-shot_litgpt lm_tinyllama no_adaptation_bf16 bias_only \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

### Lora + no calibration
python -m llmcal agnews_16_738 basic_agnews_0-shot_litgpt lm_tinyllama lora_20samples no_calibration --accelerator "gpu"
python -m llmcal agnews_16_738 basic_agnews_0-shot_litgpt lm_tinyllama lora_60samples no_calibration --accelerator "gpu"
python -m llmcal agnews_256_493 basic_agnews_0-shot_litgpt lm_tinyllama lora_1000samples no_calibration --accelerator "gpu"