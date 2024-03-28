#!/bin/bash

# Models to be downloaded and converted to dtype
declare -A MODEL_DTYPE=(
    ["TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T"]="bfloat16"
    ["TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"]="bfloat16"
    ["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]="bfloat16"
    ["meta-llama/Llama-2-7b-hf"]="bfloat16"
)

if [ -z "${LIT_CHECKPOINTS}" ]; then
    echo "Select a directory to store the checkpoints:"
    read -p ">>> " LIT_CHECKPOINTS
    echo "" >> ~/.bashrc
    echo "export LIT_CHECKPOINTS=${LIT_CHECKPOINTS}" >> ~/.bashrc
    export LIT_CHECKPOINTS=${LIT_CHECKPOINTS}
fi


for MODEL in "${!MODEL_DTYPE[@]}"; do
    DTYPE=${MODEL_DTYPE[$MODEL]}
    if [ -d "$LIT_CHECKPOINTS/$MODEL-$DTYPE" ]; then
        echo "Model $MODEL-$DTYPE already downloaded"
        continue
    fi
    litgpt download --repo_id $MODEL --dtype $DTYPE --checkpoint_dir $LIT_CHECKPOINTS
    mv $LIT_CHECKPOINTS/$MODEL $LIT_CHECKPOINTS/$MODEL-$DTYPE
    rm -f $LIT_CHECKPOINTS/$MODEL-$DTYPE/*.bin
    rm -f $LIT_CHECKPOINTS/$MODEL-$DTYPE/*.safetensors
done