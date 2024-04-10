#!/bin/bash

python scripts/main.py \
    --model tinyllama_3T_bf16 \
    --task banking77 \
    --splits all
python scripts/main.py \
    --model affine_vector_unipriors \
    --task banking77_tinyllama_3T_bf16_logits \
    --splits all \
    --model.num_classes 77
python scripts/view_results.py \
    --title "banking77 - Instruction following prompt - TinyLlama (3T) - No adaptation" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test True \
    banking77/tinyllama_3T_bf16/all
python scripts/view_results.py \
    --title "banking77 - Instruction following prompt - TinyLlama (3T) - Affine Vector" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test True \
    banking77_tinyllama_3T_bf16_logits/affine_vector_unipriors_model.num_classes=77/all

