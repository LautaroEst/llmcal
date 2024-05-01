#!/bin/bash

python scripts/main.py \
    --base_model lm_tinyllama_3T_16bf \
    --dataset sst2 \
    --prompt basic_sst2 \
    --method no_adaptation \
    --train_list n=100_rs=738
    