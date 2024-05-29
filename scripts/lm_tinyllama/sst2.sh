#!/bin/bash

### No adaptation + no calibration
python -m llmcal sst2_large basic_sst2_0-shot_litgpt lm_tinyllama no_adaptation_bf16 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1

mkdir -p experiments/sst2_small/basic_sst2_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/
ln -sf ../../../../../sst2_large/basic_sst2_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions experiments/sst2_small/basic_sst2_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache

python -m llmcal sst2_small basic_sst2_0-shot_litgpt lm_tinyllama no_adaptation_bf16 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1

mkdir -p experiments/sst2_medium/basic_sst2_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache
ln -sf ../../../../../sst2_large/basic_sst2_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions experiments/sst2_medium/basic_sst2_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache

python -m llmcal sst2_medium basic_sst2_0-shot_litgpt lm_tinyllama no_adaptation_bf16 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1

mkdir -p experiments/sst2_mini/basic_sst2_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache
ln -sf ../../../../../sst2_large/basic_sst2_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions experiments/sst2_mini/basic_sst2_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache

python -m llmcal sst2_mini basic_sst2_0-shot_litgpt lm_tinyllama no_adaptation_bf16 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1

### No adaptation + affine vector
python -m llmcal sst2_large basic_sst2_0-shot_litgpt lm_tinyllama no_adaptation_bf16 affine_vector \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

python -m llmcal sst2_small basic_sst2_0-shot_litgpt lm_tinyllama no_adaptation_bf16 affine_vector \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

python -m llmcal sst2_medium basic_sst2_0-shot_litgpt lm_tinyllama no_adaptation_bf16 affine_vector \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

python -m llmcal sst2_mini basic_sst2_0-shot_litgpt lm_tinyllama no_adaptation_bf16 affine_vector \
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
python -m llmcal sst2_large basic_sst2_0-shot_litgpt lm_tinyllama lora_v1 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1

python -m llmcal sst2_small basic_sst2_0-shot_litgpt lm_tinyllama lora_v3 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1

python -m llmcal sst2_medium basic_sst2_0-shot_litgpt lm_tinyllama lora_v4 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1

python -m llmcal sst2_mini basic_sst2_0-shot_litgpt lm_tinyllama lora_v4 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1

### Lora + affine vector
python -m llmcal sst2_large basic_sst2_0-shot_litgpt lm_tinyllama lora_v1 affine_vector \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 4e-3 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 50

python -m llmcal sst2_small basic_sst2_0-shot_litgpt lm_tinyllama lora_v3 affine_vector \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 4e-3 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 20

python -m llmcal sst2_medium basic_sst2_0-shot_litgpt lm_tinyllama lora_v4 affine_vector \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 4e-3 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 20

python -m llmcal sst2_mini basic_sst2_0-shot_litgpt lm_tinyllama lora_v4 affine_vector \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 4e-3 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 20