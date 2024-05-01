#!/bin/bash

python scripts/main.py \
    --dataset sst2 \
    --prompt prefix_basic_sst2 \
    --data_fold all_zero_shot \
    --model lm_tinyllama \
    --method lora_r=8
    
python scripts/main.py \
    --dataset sst2 \
    --prompt prefix_basic_sst2 \
    --data_fold all_8_shot_723 \
    --model lm_tinyllama \
    --method lora_r=8
