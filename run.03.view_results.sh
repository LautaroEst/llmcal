#!/bin/bash

python scripts/view_results.py \
    --title "AG News - Multiple choice prompt - TinyLlama (3T) - No adaptation" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_mc/tinyllama_3T_bf16/all

python scripts/view_results.py \
    --title "AG News - Multiple choice prompt - TinyLlama (3T) - Affine Vector" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_mc_tinyllama_3T_bf16_logits/affine_vector_model.num_classes=4/all

python scripts/view_results.py \
    --title "AG News - Multiple choice prompt - TinyLlama (3T) - LoRA r=8 lr=0.001" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_mc/tinyllama_3T_bf16_lora_train.learning_rate=0.001_model.lora_r=8/all

python scripts/view_results.py \
    --title "AG News - Multiple choice prompt - TinyLlama (3T) - LoRA r=8 lr=0.001 + Affine Vector" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_mc_tinyllama_3T_bf16_lora_logits/affine_vector_model.num_classes=4/all

python scripts/view_results.py \
    --title "AG News - Multiple choice prompt - TinyLlama (3T) - Full FT lr=0.0001" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_mc/tinyllama_3T_bf16_full_ft_train.learning_rate=0.0001/all

python scripts/view_results.py \
    --title "AG News - Multiple choice prompt - TinyLlama (3T) - Full FT + Affine Vector" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_mc_tinyllama_3T_bf16_fullft_logits/affine_vector_model.num_classes=4/all



    