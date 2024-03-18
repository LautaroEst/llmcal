#!/bin/bash

# Plot results
python scripts/plot_model_and_task_results.py \
    --task glue_sst2_inst_0-shot-AB \
    --metrics accuracy,norm_cross_entropy \
    --bootstrap 100 \
    --random_state 123

# python scripts/plot_method_vs_training_samples.py \
#     glue_sst2_inst_0-shot/tinyllama_no_adapt \
#     glue_sst2_tinyllama-logits/affine_vector \
#     --metrics accuracy,norm_cross_entropy \
#     --bootstrap 100 \
#     --random_state 123
