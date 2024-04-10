#!/bin/bash

python scripts/main.py \
    --model tinyllama_3T_bf16 \
    --task 20newsgroup \
    --splits all
python scripts/main.py \
    --model affine_vector \
    --task 20newsgroup_tinyllama_3T_bf16_logits \
    --splits all \
    --model.num_classes 20
python scripts/view_results.py \
    --title "20 newsgroup - Instruction following prompt - TinyLlama (3T) - No adaptation" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    20newsgroup/tinyllama_3T_bf16/all
python scripts/view_results.py \
    --title "20 newsgroup - Instruction following prompt - TinyLlama (3T) - Affine Vector" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    20newsgroup_tinyllama_3T_bf16_logits/affine_vector_model.num_classes=20/all

