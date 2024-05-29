#!/bin/bash

### No adaptation + no calibration
python -m llmcal 20newsgroups_large qa_20newsgroups_0-shot_litgpt phi no_adaptation_bf16 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1

mkdir -p experiments/20newsgroups_small/qa_20newsgroups_0-shot_litgpt/phi/no_adaptation_bf16/.cache/
ln -sf ../../../../../20newsgroups_large/qa_20newsgroups_0-shot_litgpt/phi/no_adaptation_bf16/.cache/predictions experiments/20newsgroups_small/qa_20newsgroups_0-shot_litgpt/phi/no_adaptation_bf16/.cache

python -m llmcal 20newsgroups_small qa_20newsgroups_0-shot_litgpt phi no_adaptation_bf16 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1

mkdir -p experiments/20newsgroups_medium/qa_20newsgroups_0-shot_litgpt/phi/no_adaptation_bf16/.cache/
ln -sf ../../../../../20newsgroups_large/qa_20newsgroups_0-shot_litgpt/phi/no_adaptation_bf16/.cache/predictions experiments/20newsgroups_medium/qa_20newsgroups_0-shot_litgpt/phi/no_adaptation_bf16/.cache

python -m llmcal 20newsgroups_medium qa_20newsgroups_0-shot_litgpt phi no_adaptation_bf16 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1

mkdir -p experiments/20newsgroups_mini/qa_20newsgroups_0-shot_litgpt/phi/no_adaptation_bf16/.cache/
ln -sf ../../../../../20newsgroups_large/qa_20newsgroups_0-shot_litgpt/phi/no_adaptation_bf16/.cache/predictions experiments/20newsgroups_mini/qa_20newsgroups_0-shot_litgpt/phi/no_adaptation_bf16/.cache

python -m llmcal 20newsgroups_mini qa_20newsgroups_0-shot_litgpt phi no_adaptation_bf16 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1

### No adaptation + affine vector
python -m llmcal 20newsgroups_large qa_20newsgroups_0-shot_litgpt phi no_adaptation_bf16 affine_vector \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-4 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 400

python -m llmcal 20newsgroups_small qa_20newsgroups_0-shot_litgpt phi no_adaptation_bf16 affine_vector \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

python -m llmcal 20newsgroups_medium qa_20newsgroups_0-shot_litgpt phi no_adaptation_bf16 affine_vector \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

python -m llmcal 20newsgroups_mini qa_20newsgroups_0-shot_litgpt phi no_adaptation_bf16 affine_vector \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

### Lora + no calibration
python -m llmcal 20newsgroups_large qa_20newsgroups_0-shot_litgpt phi lora_v2 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1

python -m llmcal 20newsgroups_small qa_20newsgroups_0-shot_litgpt phi lora_v3 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1

python -m llmcal 20newsgroups_medium qa_20newsgroups_0-shot_litgpt phi lora_v4 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1

python -m llmcal 20newsgroups_mini qa_20newsgroups_0-shot_litgpt phi lora_v4 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1

### Lora + affine vector
python -m llmcal 20newsgroups_large qa_20newsgroups_0-shot_litgpt phi lora_v2 affine_vector \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 5e-3 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 40

python -m llmcal 20newsgroups_small qa_20newsgroups_0-shot_litgpt phi lora_v3 affine_vector \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 4e-3 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 20

python -m llmcal 20newsgroups_medium qa_20newsgroups_0-shot_litgpt phi lora_v4 affine_vector \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 4e-3 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 20

python -m llmcal 20newsgroups_mini qa_20newsgroups_0-shot_litgpt phi lora_v4 affine_vector \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 4e-3 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 20
