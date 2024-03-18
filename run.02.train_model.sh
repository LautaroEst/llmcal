#!/bin/bash

# python scripts/main.py \
#     --model tinyllama_no_adapt \
#     --task glue_sst2_inst_0-shot \
#     --splits all

# python scripts/main.py \
#     --model tinyllama_no_adapt \
#     --task glue_sst2_inst_4-shot \
#     --splits n=100_rs=7384

python scripts/main.py \
    --model affine_vector \
    --task glue_sst2_tinyllama-logits \
    --splits n=10_rs=89