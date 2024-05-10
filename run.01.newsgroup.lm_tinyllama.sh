#!/bin/bash

MODEL=lm_tinyllama
DATA_FOLD=n=10000_0-shot_6273

python scripts/main.py \
    --dataset 20newsgroup \
    --prompt prefix_basic_20newsgroup \
    --data_fold $DATA_FOLD \
    --model $MODEL \
    --method no_adaptation \

python scripts/main.py \
    --dataset 20newsgroup \
    --prompt prefix_basic_20newsgroup \
    --data_fold $DATA_FOLD \
    --model $MODEL \
    --method affine_vector

python scripts/main.py \
    --dataset 20newsgroup \
    --prompt prefix_basic_20newsgroup \
    --data_fold $DATA_FOLD \
    --model $MODEL \
    --method lora_r=8 \
    --method.learning_rate 0.0001 \
    --method.max_epochs 6

# python scripts/main.py \
#     --dataset 20newsgroup \
#     --prompt prefix_basic_20newsgroup \
#     --data_fold $DATA_FOLD \
#     --model $MODEL \
#     --method full_ft