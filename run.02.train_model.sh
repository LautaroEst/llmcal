#!/bin/bash

python scripts/main.py \
    --model tinyllama_3T_bf16 \
    --task tony_zhao_agnews_mc \
    --splits all

python scripts/main.py \
    --model affine_vector \
    --task tony_zhao_agnews_mc_tinyllama_3T_bf16_logits \
    --splits all \
    --model.num_classes 4

python scripts/main.py \
    --model tinyllama_3T_bf16_lora \
    --task tony_zhao_agnews_mc \
    --splits all \
    --train.learning_rate 0.0001