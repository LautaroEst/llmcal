#!/bin/bash -ex

source ./scripts/env.sh


run_test() {
    local model=$1
    local dataset=$2
    local precision="bf16-true"

    # Predictions directories and lists
    local model_dir="$CHECKPOINTS_DIR/${model2checkpoint[$model]}"
    local data_path=outputs/prompts/$model/$dataset/all.jsonl
    local test_list="test_${dataset2testsize[$dataset]}"
    local output_dir="outputs/no_adaptation/$model/$dataset/size=all/seed=all/test=$dataset/list=$test_list"
    if [ ! -f $output_dir/logits.csv ]; then
        mkdir -p $output_dir
        python -m llmcal.scripts.run_posteriors \
            --base_checkpoint_dir $model_dir \
            --checkpoint_dir $model_dir \
            --data_path $data_path \
            --output_dir $output_dir \
            --prediction_lists lists/$dataset/$test_list.txt \
            --precision $precision \
            --devices 1 \
            --num_nodes 1 \
            --batch_size 1 \
            --max_seq_length $max_seq_length
    fi
}

run_train() {
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
}


# 1: model
# 2: sizes
# 3: val_check_interval
run_all() {
    local model=$1
    for size in ${FACTORS[@]}; do
        for dataset in "${DATASETS[@]}"; do
            local test_list="test_${dataset2testsize[$dataset]}"
            local num_seeds=${dataset2nseeds[$dataset]}
            for num_seed in $(seq 0 $(($num_seeds - 1))); do
                run_test $model $dataset
                run_train $model $dataset $size $num_seed
            done
        done
    done
}

run_all $model
