#!/bin/bash

python -m llmcal medical-abstracts_large instr_medical-abstracts_0-shot_litgpt lm_tinyllama_chat no_adaptation_bf16 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1

python -m llmcal medical-abstracts_large instr_medical-abstracts_0-shot_litgpt lm_tinyllama_chat no_adaptation_bf16 affine_vector \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-4 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 400

python -m llmcal medical-abstracts_large instr_medical-abstracts_0-shot_litgpt lm_tinyllama_chat lora_v2 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1

python -m llmcal medical-abstracts_large instr_medical-abstracts_0-shot_litgpt lm_tinyllama_chat lora_v2 affine_vector \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 5e-3 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 40
