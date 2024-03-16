#!/bin/bash

python scripts/main.py \
    --model tinyllama_no_adapt \
    --task glue--sst2_inst_4-shot \
    --splits n=100_rs=7384

# python scripts/main.py \
#     --model affine_vector \
#     --task glue--sst2_tinyllama-logits \
#     --splits n=10_rs=89