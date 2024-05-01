#!/bin/bash

# Models to be downloaded and converted to dtype
declare -a MODELS=(
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
)

if [ -z "${LIT_CHECKPOINTS}" ]; then
    echo "Select a directory to store the checkpoints:"
    read -p ">>> " LIT_CHECKPOINTS
    echo "" >> ~/.bashrc
    echo "export LIT_CHECKPOINTS=${LIT_CHECKPOINTS}" >> ~/.bashrc
    export LIT_CHECKPOINTS=${LIT_CHECKPOINTS}
fi


for MODEL in "$MODELS"; do
    if [ -d "$LIT_CHECKPOINTS/$MODEL" ]; then
        echo "Model $MODEL already downloaded"
        continue
    fi
    litgpt download --repo_id $MODEL --checkpoint_dir $LIT_CHECKPOINTS
    rm -f $LIT_CHECKPOINTS/$MODEL/*.bin
    rm -f $LIT_CHECKPOINTS/$MODEL/*.safetensors
done