#!/bin/bash

### No adaptation + no calibration
python -m llmcal 20newsgroups_4_295 basic_20newsgroups_0-shot_litgpt lm_phi3 no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

mkdir -p experiments/20newsgroups_16_738/basic_20newsgroups_0-shot_litgpt/lm_phi3/no_adaptation_bf16/.cache/predictions
ln -sf ../../../../../../20newsgroups_4_295/basic_20newsgroups_0-shot_litgpt/lm_phi3/no_adaptation_bf16/.cache/predictions/test experiments/20newsgroups_16_738/basic_20newsgroups_0-shot_litgpt/lm_phi3/no_adaptation_bf16/.cache/predictions
python -m llmcal 20newsgroups_16_738 basic_20newsgroups_0-shot_litgpt lm_phi3 no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

mkdir -p experiments/20newsgroups_256_493/basic_20newsgroups_0-shot_litgpt/lm_phi3/no_adaptation_bf16/.cache/predictions
ln -sf ../../../../../../20newsgroups_4_295/basic_20newsgroups_0-shot_litgpt/lm_phi3/no_adaptation_bf16/.cache/predictions/test experiments/20newsgroups_256_493/basic_20newsgroups_0-shot_litgpt/lm_phi3/no_adaptation_bf16/.cache/predictions
python -m llmcal 20newsgroups_256_493 basic_20newsgroups_0-shot_litgpt lm_phi3 no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

mkdir -p experiments/20newsgroups_2_927/basic_20newsgroups_0-shot_litgpt/lm_phi3/no_adaptation_bf16/.cache/predictions
ln -sf ../../../../../../20newsgroups_4_295/basic_20newsgroups_0-shot_litgpt/lm_phi3/no_adaptation_bf16/.cache/predictions/test experiments/20newsgroups_2_927/basic_20newsgroups_0-shot_litgpt/lm_phi3/no_adaptation_bf16/.cache/predictions
python -m llmcal 20newsgroups_2_927 basic_20newsgroups_0-shot_litgpt lm_phi3 no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

### No adaptation + affine scalar
python -m llmcal 20newsgroups_2_927 basic_20newsgroups_0-shot_litgpt lm_phi3 no_adaptation_bf16 affine_scalar \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

python -m llmcal 20newsgroups_4_295 basic_20newsgroups_0-shot_litgpt lm_phi3 no_adaptation_bf16 affine_scalar \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

python -m llmcal 20newsgroups_16_738 basic_20newsgroups_0-shot_litgpt lm_phi3 no_adaptation_bf16 affine_scalar \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

python -m llmcal 20newsgroups_256_493 basic_20newsgroups_0-shot_litgpt lm_phi3 no_adaptation_bf16 affine_scalar \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

### No adaptation + temp scaling
python -m llmcal 20newsgroups_2_927 basic_20newsgroups_0-shot_litgpt lm_phi3 no_adaptation_bf16 temp_scaling \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

python -m llmcal 20newsgroups_4_295 basic_20newsgroups_0-shot_litgpt lm_phi3 no_adaptation_bf16 temp_scaling \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

python -m llmcal 20newsgroups_16_738 basic_20newsgroups_0-shot_litgpt lm_phi3 no_adaptation_bf16 temp_scaling \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

python -m llmcal 20newsgroups_256_493 basic_20newsgroups_0-shot_litgpt lm_phi3 no_adaptation_bf16 temp_scaling \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

### No adaptation + bias only
python -m llmcal 20newsgroups_2_927 basic_20newsgroups_0-shot_litgpt lm_phi3 no_adaptation_bf16 bias_only \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

python -m llmcal 20newsgroups_4_295 basic_20newsgroups_0-shot_litgpt lm_phi3 no_adaptation_bf16 bias_only \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

python -m llmcal 20newsgroups_16_738 basic_20newsgroups_0-shot_litgpt lm_phi3 no_adaptation_bf16 bias_only \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

python -m llmcal 20newsgroups_256_493 basic_20newsgroups_0-shot_litgpt lm_phi3 no_adaptation_bf16 bias_only \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

### Lora + no calibration
python -m llmcal 20newsgroups_2_927 basic_20newsgroups_0-shot_litgpt lm_phi3 lora_40samples no_calibration --accelerator "gpu"
python -m llmcal 20newsgroups_4_295 basic_20newsgroups_0-shot_litgpt lm_phi3 lora_60samples no_calibration --accelerator "gpu"
python -m llmcal 20newsgroups_16_738 basic_20newsgroups_0-shot_litgpt lm_phi3 lora_200samples no_calibration --accelerator "gpu"
python -m llmcal 20newsgroups_256_493 basic_20newsgroups_0-shot_litgpt lm_phi3 lora_3500samples no_calibration --accelerator "gpu"