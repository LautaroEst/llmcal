#!/bin/bash

# Directories:
SCRIPTS_DIR="./scripts"
RESULTS_DIR="./results"

# Model:
MODEL="gpt2-xl"
device="gpu"

# Template
TEMPLATE="0_shot"

# Dataset:
declare -A DATASETS=(
    ["tony_zhao/trec"]="6"
    ["tony_zhao/sst2"]="2"
    ["tony_zhao/agnews"]="4"
    ["tony_zhao/dbpedia"]="14"
)

# Hyperparameters sets:
weight_decay=(0.0 0.00001 0.0001 0.001 0.01)

# Tuning:
# dataset="tony_zhao/trec"
dataset="tony_zhao/sst2"

# Output directory:
OUTPUT_DIR="$RESULTS_DIR/tune_calibration/$MODEL/$dataset/"

# Number of train samples:
num_train_samples=3800

# Run the script:
for wd in "${weight_decay[@]}"; do
    python $SCRIPTS_DIR/calibrate_features.py \
        --train_features "$RESULTS_DIR/run_dataset_on_model/$MODEL/$dataset/train/$TEMPLATE/logits.npy" \
        --train_labels "$RESULTS_DIR/run_dataset_on_model/$MODEL/$dataset/train/$TEMPLATE/labels.npy" \
        --subsample_train $num_train_samples \
        --validation_samples 200 \
        --num_classes "${DATASETS[$dataset]}" \
        --method "affine" \
        --feature_map "identity" \
        --alpha "matrix" \
        --bias \
        --accelerator "$device" \
        --devices 1 \
        --loss "log-loss" \
        --learning_rate 0.01 \
        --max_epochs 1000 \
        --weight_decay $wd \
        --batch_size 32 \
        --max_ls 40 \
        --output_dir "$OUTPUT_DIR/$TEMPLATE--wd=$wd--num_samples=$num_train_samples" \
        --random_state 8394
done