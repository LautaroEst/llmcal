#!/bin/bash

python scripts/view_results.py \
    --title "AG News - Multiple choice prompt - TinyLlama (3T) - No adaptation" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_mc/tinyllama/all

python scripts/view_results.py \
    --title "AG News - Multiple choice prompt - TinyLlama (3T) - Affine Matrix" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_mc_tinyllama_logits/affine_matrix/all

python scripts/view_results.py \
    --title "MNLI - Multiple choice prompt - TinyLlama (3T) - No adaptation" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    glue_mnli_mc/tinyllama/all

python scripts/view_results.py \
    --title "MNLI - Multiple choice prompt - TinyLlama (3T) - Affine Matrix" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    glue_mnli_mc_tinyllama_logits/affine_matrix/all

python scripts/view_results.py \
    --title "SST-2 (GLUE) - Multiple choice prompt - TinyLlama (3T) - No adaptation" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    glue_sst2_mc/tinyllama/all

python scripts/view_results.py \
    --title "SST-2 (GLUE) - Multiple choice prompt - TinyLlama (3T) - Affine Matrix" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    glue_sst2_mc_tinyllama_logits/affine_matrix/all


    