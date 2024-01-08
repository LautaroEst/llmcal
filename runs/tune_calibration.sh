#!/bin/bash

# Directories:
SCRIPTS_DIR="./scripts"
RESULTS_DIR="./results"

# Model:
MODEL="gpt2-xl"

# Template
TEMPLATE="0_shot"

# Dataset:
declare -A DATASETS=(
    ["tony_zhao/trec"]="6"
    ["tony_zhao/sst2"]="2"
    ["tony_zhao/agnews"]="4"
    ["tony_zhao/dbpedia"]="14"
)

# Tuning:
dataset="tony_zhao/trec"
python $SCRIPTS_DIR/calibrate_features.py \
    --train_features "$RESULTS_DIR/run_dataset_on_model/$MODEL/$dataset/train/$TEMPLATE/logits.npy" \
    --train_labels "$RESULTS_DIR/run_dataset_on_model/$MODEL/$dataset/train/$TEMPLATE/labels.npy" \
    --subsample_train 3800 \
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
    --max_epochs 100 \
    --batch_size 32 \
    --max_ls 40 \
    --tolerance 0.00001 \
    --output_dir "$RESULTS_DIR/tune_calibration/$MODEL/$dataset/$TEMPLATE/" \
    --random_state 8394

