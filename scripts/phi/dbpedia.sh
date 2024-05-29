#!/bin/bash

### No adaptation + no calibration
python -m llmcal dbpedia_large qa_dbpedia_0-shot_litgpt phi no_adaptation_bf16 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1

mkdir -p experiments/dbpedia_small/qa_dbpedia_0-shot_litgpt/phi/no_adaptation_bf16/.cache/
ln -sf ../../../../../dbpedia_large/qa_dbpedia_0-shot_litgpt/phi/no_adaptation_bf16/.cache/predictions experiments/dbpedia_small/qa_dbpedia_0-shot_litgpt/phi/no_adaptation_bf16/.cache

python -m llmcal dbpedia_small qa_dbpedia_0-shot_litgpt phi no_adaptation_bf16 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1

mkdir -p experiments/dbpedia_medium/qa_dbpedia_0-shot_litgpt/phi/no_adaptation_bf16/.cache
ln -sf ../../../../../dbpedia_large/qa_dbpedia_0-shot_litgpt/phi/no_adaptation_bf16/.cache/predictions experiments/dbpedia_medium/qa_dbpedia_0-shot_litgpt/phi/no_adaptation_bf16/.cache

python -m llmcal dbpedia_medium qa_dbpedia_0-shot_litgpt phi no_adaptation_bf16 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1

mkdir -p experiments/dbpedia_mini/qa_dbpedia_0-shot_litgpt/phi/no_adaptation_bf16/.cache
ln -sf ../../../../../dbpedia_large/qa_dbpedia_0-shot_litgpt/phi/no_adaptation_bf16/.cache/predictions experiments/dbpedia_mini/qa_dbpedia_0-shot_litgpt/phi/no_adaptation_bf16/.cache

python -m llmcal dbpedia_mini qa_dbpedia_0-shot_litgpt phi no_adaptation_bf16 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1

### No adaptation + affine vector
python -m llmcal dbpedia_large qa_dbpedia_0-shot_litgpt phi no_adaptation_bf16 affine_vector \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

python -m llmcal dbpedia_small qa_dbpedia_0-shot_litgpt phi no_adaptation_bf16 affine_vector \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

python -m llmcal dbpedia_medium qa_dbpedia_0-shot_litgpt phi no_adaptation_bf16 affine_vector \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

python -m llmcal dbpedia_mini qa_dbpedia_0-shot_litgpt phi no_adaptation_bf16 affine_vector \
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
python -m llmcal dbpedia_large qa_dbpedia_0-shot_litgpt phi lora_v1 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1

python -m llmcal dbpedia_small qa_dbpedia_0-shot_litgpt phi lora_v3 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1

python -m llmcal dbpedia_medium qa_dbpedia_0-shot_litgpt phi lora_v4 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1

python -m llmcal dbpedia_mini qa_dbpedia_0-shot_litgpt phi lora_v4 no_calibration \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1

### Lora + affine vector
python -m llmcal dbpedia_large qa_dbpedia_0-shot_litgpt phi lora_v1 affine_vector \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 4e-3 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 50

python -m llmcal dbpedia_small qa_dbpedia_0-shot_litgpt phi lora_v3 affine_vector \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 4e-3 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 20

python -m llmcal dbpedia_medium qa_dbpedia_0-shot_litgpt phi lora_v4 affine_vector \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 4e-3 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 20

python -m llmcal dbpedia_mini qa_dbpedia_0-shot_litgpt phi lora_v4 affine_vector \
    --accelerator "gpu" \
    --strategy "auto" \
    --devices 1 \
    --num_nodes 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 4e-3 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 20