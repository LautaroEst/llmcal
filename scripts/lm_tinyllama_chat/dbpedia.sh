#!/bin/bash

python -m llmcal dbpedia_large instr_dbpedia_0-shot_litgpt lm_tinyllama_chat no_adaptation_bf16 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1

python -m llmcal dbpedia_large instr_dbpedia_0-shot_litgpt lm_tinyllama_chat no_adaptation_bf16 affine_vector \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

python -m llmcal dbpedia_large instr_dbpedia_0-shot_litgpt lm_tinyllama_chat lora_v1 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1

python -m llmcal dbpedia_large instr_dbpedia_0-shot_litgpt lm_tinyllama_chat lora_v1 affine_vector \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 20
