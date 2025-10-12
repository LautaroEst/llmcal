#!/bin/bash -ex

source ./scripts/env.sh

run_dirichlet_calibration() {
    local model=$1
    local dataset=$2
    local size=$3
    local num_seed=$4
    local method=$5
    local precision="bf16-true"
    local seed=$((base_seed+num_seed))
    local learning_rate=1e-3
    local tolerance=1e-5
    local max_ls=40
    local model_dir="$CHECKPOINTS_DIR/${model2checkpoint[$model]}"
    local data_path="outputs/prompts/$model/$dataset/all.jsonl"
    local test_list="test_${dataset2testsize[$dataset]}"
    local train_list="0.0-1.0"

    # Predictions directories and lists
    local prediction_dir="outputs/no_adaptation/$model/$dataset/size=$size/seed=$num_seed/test=$dataset/list=$train_list"
    if [ ! -f $prediction_dir/logits.csv ]; then
        mkdir -p $prediction_dir
        python -m llmcal.scripts.run_posteriors \
            --base_checkpoint_dir $model_dir \
            --checkpoint_dir $model_dir \
            --data_path $data_path \
            --output_dir $prediction_dir \
            --prediction_lists lists/$dataset/size=$size/seed=$num_seed/$train_list.txt \
            --precision $precision \
            --devices 1 \
            --num_nodes 1 \
            --batch_size 1 \
            --max_seq_length $max_seq_length
    fi

    # Calibration directories
    cal_dir="outputs/calibration/$model/$dataset/size=$size/seed=$num_seed/$method/$train_list/$train_list"
    if [ ! -f "$cal_dir/test=$dataset/list=$test_list/logits.csv" ]; then
        mkdir -p $cal_dir/test=$dataset/list=$test_list $cal_dir/logs
        python -m llmcal.scripts.dirichlet_cal \
            --output_dir $cal_dir/test=$dataset/list=$test_list \
            --log_dir $cal_dir/logs \
            --checkpoint_dir $cal_dir \
            --train_logits $prediction_dir/logits.csv \
            --train_labels $prediction_dir/labels.csv \
            --predict_logits "outputs/no_adaptation/$model/$dataset/size=all/seed=all/test=$dataset/list=$test_list/logits.csv" \
            --predict_labels "outputs/no_adaptation/$model/$dataset/size=all/seed=all/test=$dataset/list=$test_list/labels.csv" \
            --method $method \
            --seed $seed
    fi
}


# 1: model
# 2: sizes
# 3: val_check_interval
run_cal_vs_samples() {
    local model=$1
    for size in ${FACTORS[@]}; do
        for dataset in "${DATASETS[@]}"; do
            local test_list="test_${dataset2testsize[$dataset]}"
            local num_seeds=${dataset2nseeds[$dataset]}
            for num_seed in $(seq 0 $(($num_seeds - 1))); do

                run_dirichlet_calibration $model $dataset $size $num_seed "dirichlet_fixed_diag"
                run_dirichlet_calibration $model $dataset $size $num_seed "dirichlet_full_l2"
                # run_dirichlet_calibration $model $dataset $size $num_seed "dirichlet_odir_l2"
            done
        done
    done
}

run_cal_vs_samples $model 16

