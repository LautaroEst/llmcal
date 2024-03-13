#!/bin/bash

# python scripts/main.py \
#     --model tinyllama \
#     --train_task glue--sst2_inst_4-shot \
#     --test_task glue--sst2_inst_4-shot \
#     --splits n=100_rs=7384

python scripts/main.py \
    --model affine_vector \
    --train_task glue--sst2_tinyllama-logits \
    --test_task glue--sst2_tinyllama-logits \
    --splits n=10_rs=89