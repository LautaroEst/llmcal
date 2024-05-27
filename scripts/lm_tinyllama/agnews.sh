#!/bin/bash

### No adaptation + no calibration
python -m llmcal agnews_large basic_agnews_0-shot_litgpt lm_tinyllama no_adaptation_bf16 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1

mkdir -p experiments/agnews_small/basic_agnews_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/no_calibration/
ln -sf ../../../../../agnews_large/basic_agnews_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/no_calibration/predictions \
    experiments/agnews_small/basic_agnews_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/no_calibration

mkdir -p experiments/agnews_medium/basic_agnews_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/no_calibration/
ln -sf ../../../../../agnews_large/basic_agnews_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/no_calibration/predictions \
    experiments/agnews_medium/basic_agnews_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/no_calibration

### No adaptation + affine vector
python -m llmcal agnews_large basic_agnews_0-shot_litgpt lm_tinyllama no_adaptation_bf16 affine_vector \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-4 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 400

### Lora + no calibration
python -m llmcal agnews_large basic_agnews_0-shot_litgpt lm_tinyllama lora_v2 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1

### Lora + affine vector
python -m llmcal agnews_large basic_agnews_0-shot_litgpt lm_tinyllama lora_v2 affine_vector \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 5e-3 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 40
