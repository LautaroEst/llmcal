#!/bin/bash

# Plot results
python scripts/plot_method_vs_training_samples.py \
    glue_sst2_inst_0-shot-AB_tinyllama-logits/no_adapt \
    glue_sst2_inst_0-shot-AB_tinyllama-logits/affine_vector \
    --baseline_method glue_sst2_inst_0-shot-AB_prompt/tinyllama/all \
    --metrics norm_cross_entropy,error_rate \
    --bootstrap 100 \
    --random_state 123
