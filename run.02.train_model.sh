#!/bin/bash

python scripts/train_model.py \
    --dataset tony_zhao--agnews_n=10_rs=89 \
    --model tinyllama \
    --prompt news_0-shot