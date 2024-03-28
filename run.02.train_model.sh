#!/bin/bash

# Full-FT
python scripts/main.py \
    --model tinyllama_full_ft \
    --task tony_zhao_agnews_mc \
    --splits n=100_rs=7384