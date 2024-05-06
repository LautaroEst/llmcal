#!/bin/bash

MODEL=lm_tinyllama
DATA_FOLD=n=10000_0-shot_6273

python scripts/main.py \
    --dataset sst2 \
    --prompt prefix_basic_sst2 \
    --data_fold $DATA_FOLD \
    --model $MODEL \
    --method no_adaptation

python scripts/main.py \
    --dataset sst2 \
    --prompt prefix_basic_sst2 \
    --data_fold $DATA_FOLD \
    --model $MODEL \
    --method affine_vector

python scripts/main.py \
    --dataset sst2 \
    --prompt prefix_basic_sst2 \
    --data_fold $DATA_FOLD \
    --model $MODEL \
    --method lora_r=8

python scripts/main.py \
    --dataset sst2 \
    --prompt prefix_basic_sst2 \
    --data_fold $DATA_FOLD \
    --model $MODEL \
    --method full_ft