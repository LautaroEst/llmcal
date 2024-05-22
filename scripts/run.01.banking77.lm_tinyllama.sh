#!/bin/bash

python -m llmcal banking77_large basic_banking77_0-shot_litgpt lm_tinyllama no_adaptation_bf16 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1

python -m llmcal banking77_large basic_banking77_0-shot_litgpt lm_tinyllama no_adaptation_bf16 affine_vector \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 40

python -m llmcal banking77_large basic_banking77_0-shot_litgpt lm_tinyllama lora_v2 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1

python -m llmcal banking77_large basic_banking77_0-shot_litgpt lm_tinyllama lora_v2 affine_vector \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 40
