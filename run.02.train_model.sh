#!/bin/bash

# Pre-compute the logits through the model
# python scripts/main.py \
#     --model tinyllama \
#     --task glue_sst2_inst_0-shot-AB_prompt \
#     --splits all

# for split in n=10_rs=89 n=100_rs=7384; do
#     for method in affine_vector; do
#         python scripts/main.py \
#             --model $method \
#             --task glue_sst2_inst_0-shot-AB_tinyllama-logits \
#             --splits $split
#     done
# done

# python scripts/main.py \
#     --model tinyllama \
#     --task refind_inst_0-shot_prompt \
#     --splits all

# python scripts/main.py \
#     --model tinyllama \
#     --task tony_zhao_dbpedia_inst_0-shot_prompt \
#     --splits all

# python scripts/main.py \
#     --model tinyllama \
#     --task tony_zhao_agnews_inst_0-shot_prompt \
#     --splits all

# python scripts/main.py \
#     --model tinyllama_full_ft \
#     --task glue_sst2_mc \
#     --splits n=100_rs=7384



## Affine methods:
python scripts/main.py \
    --model tinyllama \
    --task glue_sst2_orig \
    --splits all
# python scripts/main.py \
#     --model affine_vector \
#     --task glue_sst2_mc \
#     --splits n=100_rs=7384