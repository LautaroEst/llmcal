#!/bin/bash

MODEL=lm_tinyllama
DATA_FOLD=n=10000_0-shot_6273

python scripts/main.py \
    --dataset medical_abstracts \
    --prompt prefix_basic_medical_abstracts \
    --data_fold $DATA_FOLD \
    --model $MODEL \
    --method no_adaptation \

python scripts/main.py \
    --dataset medical_abstracts \
    --prompt prefix_basic_medical_abstracts \
    --data_fold $DATA_FOLD \
    --model $MODEL \
    --method affine_vector \
    --method.learning_rate 0.001 \
    --method.max_epochs 100

python scripts/main.py \
    --dataset medical_abstracts \
    --prompt prefix_basic_medical_abstracts \
    --data_fold $DATA_FOLD \
    --model $MODEL \
    --method lora_r=8 \
    --method.learning_rate 0.0001 \
    --method.max_epochs 5

# python scripts/main.py \
#     --dataset medical_abstracts \
#     --prompt prefix_basic_medical_abstracts \
#     --data_fold $DATA_FOLD \
#     --model $MODEL \
#     --method full_ft