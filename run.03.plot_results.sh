#!/bin/bash

# Plot results
# python scripts/plot_method_vs_training_samples.py \
#     glue_sst2_inst_0-shot-AB_tinyllama-logits/no_adapt \
#     glue_sst2_inst_0-shot-AB_tinyllama-logits/affine_vector \
#     --baseline_method glue_sst2_inst_0-shot-AB_prompt/tinyllama/all \
#     --metrics norm_cross_entropy,error_rate \
#     --bootstrap 100 \
#     --random_state 123

python scripts/plot_method_vs_training_samples.py \
    glue_sst2_mc/tinyllama_affine_vector \
    glue_sst2_mc/tinyllama_affine_temp_scaling \
    glue_sst2_mc/tinyllama_affine_bias \
    glue_sst2_mc/tinyllama_lora \
    glue_sst2_mc/tinyllama_full \
    glue_sst2_mc/tinyllama_adapters \
    glue_sst2/tinyllama_clf \
    --baseline_method glue_sst2_mc/tinyllama/all \
    --metrics norm_cross_entropy,error_rate,f1_score \
    --bootstrap 100 \
    --random_state 123
