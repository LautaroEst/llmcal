#!/bin/bash -ex

CHECKPOINTS_DIR=./checkpoints
HF_TOKEN=$(cat hf_token.txt)
model="llama3.2-1b-instruct"
# model="qwen2.5-7b-instruct"

# Reproducibility
base_seed=2834
declare -A dataset2nseeds=(
    ["sst2"]=5
    ["agnews"]=5
    ["dbpedia"]=5
    ["20newsgroups"]=5
    ["banking77"]=5
)

# Supported models
declare -A model2checkpoint=(
    ["llama3.2-1b"]="meta-llama/Llama-3.2-1B"
    ["llama3.2-1b-instruct"]="meta-llama/Llama-3.2-1B-Instruct"
    ["qwen2.5-7b"]="Qwen/Qwen2.5-7B"
    ["qwen2.5-7b-instruct"]="Qwen/Qwen2.5-7B-Instruct"
)

# Datasets
declare -a DATASETS=(sst2 agnews dbpedia 20newsgroups banking77)
# declare -a DATASETS=(agnews dbpedia 20newsgroups)

# Train sizes
# declare -a FACTORS=(16 32 64 128 256)
declare -a FACTORS=(16 256)
# declare -a FACTORS=(256 )

# Test sizes
declare -A dataset2testsize=(
    ["sst2"]=400
    ["agnews"]=400
    ["dbpedia"]=700
    ["20newsgroups"]=800
    ["banking77"]=1000
)

max_seq_length=2048
inference_max_seq_len=20000

export CUDA_VISIBLE_DEVICES=0

run_no_adaptation() {
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

run_affine_calibration() {
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
        python -m llmcal.scripts.affine_calibration \
            --output_dir $cal_dir/test=$dataset/list=$test_list \
            --log_dir $cal_dir/logs \
            --checkpoint_dir $cal_dir \
            --train_logits $prediction_dir/logits.csv \
            --train_labels $prediction_dir/labels.csv \
            --predict_logits "outputs/no_adaptation/$model/$dataset/size=all/seed=all/test=$dataset/list=$test_list/logits.csv" \
            --predict_labels "outputs/no_adaptation/$model/$dataset/size=all/seed=all/test=$dataset/list=$test_list/labels.csv" \
            --method $method \
            --learning_rate $learning_rate \
            --tolerance $tolerance \
            --max_ls $max_ls \
            --seed $seed
    fi
}


run_adats_calibration() {
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
        python -m llmcal.scripts.adats_cal \
            --output_dir $cal_dir/test=$dataset/list=$test_list \
            --log_dir $cal_dir/logs \
            --checkpoint_dir $cal_dir \
            --train_logits $prediction_dir/logits.csv \
            --train_embeddings $prediction_dir/embeddings.csv \
            --train_labels $prediction_dir/labels.csv \
            --predict_logits "outputs/no_adaptation/$model/$dataset/size=all/seed=all/test=$dataset/list=$test_list/logits.csv" \
            --predict_embeddings "outputs/no_adaptation/$model/$dataset/size=all/seed=all/test=$dataset/list=$test_list/embeddings.csv" \
            --predict_labels "outputs/no_adaptation/$model/$dataset/size=all/seed=all/test=$dataset/list=$test_list/labels.csv" \
            --method $method \
            --seed $seed
    fi
}


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

                # Run baseline
                run_no_adaptation $model $dataset

                # Run affine calibration
                for method in "dp_calibration" "temp_scaling" "vector_scaling" "bias_shift"; do
                    run_affine_calibration $model $dataset $size $num_seed $method
                done

                # Run adats calibration
                run_adats_calibration $model $dataset $size $num_seed "adats_z_16"

                # Run Dirichlet calibration
                for method in "dirichlet_fixed_diag" "dirichlet_full_l2"; do
                    run_dirichlet_calibration $model $dataset $size $num_seed $method
                done
            done
        done
    done
}

run_cal_vs_samples $model 16

