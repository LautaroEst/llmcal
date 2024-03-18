#!/bin/bash

if [ -z "${LIT_CHECKPOINTS}" ]; then
    echo "Select a directory to store the checkpoints:"
    read -p ">>> " LIT_CHECKPOINTS
    echo "" >> ~/.bashrc
    echo "export LIT_CHECKPOINTS=${LIT_CHECKPOINTS}" >> ~/.bashrc
    export LIT_CHECKPOINTS=${LIT_CHECKPOINTS}
fi

python scripts/download_model_and_convert_to_lit.py \
    --repo_id TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --dtype bfloat16

python scripts/download_model_and_convert_to_lit.py \
    --repo_id TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T \
    --dtype bfloat16

python scripts/download_model_and_convert_to_lit.py \
    --repo_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --dtype bfloat16

python scripts/download_model_and_convert_to_lit.py \
    --repo_id meta-llama/Llama-2-7b-hf \
    --dtype bfloat16
