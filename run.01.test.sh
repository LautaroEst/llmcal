#!/bin/bash

python scripts/main.py \
    --base_model lm_tinyllama_3T_16bf \
    --dataset 20newsgroup \
    --prompt basic_20newsgroup \
    --method no_adaptation \
    --fold all
    