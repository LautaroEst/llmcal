#!/bin/bash

# Plot results
python scripts/plot_model_and_task_results.py \
    --model affine_vector \
    --train_task glue--sst2_tinyllama-logits \
    --test_task glue--sst2_tinyllama-logits \
    --metrics accuracy \
    --bootstrap 100 \
    --random_state 123