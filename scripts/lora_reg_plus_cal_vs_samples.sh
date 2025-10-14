#!/bin/bash

CHECKPOINTS_DIR=./checkpoints
HF_TOKEN=$(cat hf_token.txt)
model="llama3.2-1b-instruct"
# model="qwen2.5-7b-instruct"

# Reproducibility
base_seed=2834
declare -A dataset2nseeds=(
    ["sst2"]=9
    ["agnews"]=9
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
# declare -a DATASETS=(sst2 agnews dbpedia 20newsgroups banking77)
# declare -a DATASETS=(sst2 banking77)
# declare -a DATASETS=(agnews dbpedia 20newsgroups)
declare -a DATASETS=(agnews )

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

# 1: model
# 2: dataset
# 3: size
# 4: loss
# 5: num_seed
# 6: val_check_interval
# 7: train_dir
# 8: early_stopping
# 9: train_list
# 10: val_list
# 11: test_list
run_lora() {
    local model=$1
    local dataset=$2
    local size=$3
    local loss=$4
    local num_seed=$5
    local seed=$((base_seed + num_seed))
    local val_check_interval=$6
    local train_dir=$7
    local early_stopping=$8
    local train_list=$9
    local val_list=${10}
    local test_list=${11}
    local lora_args="--lora_r=8 --lora_alpha=16 --lora_dropout=0.05 --lora_query --lora_key --lora_value --lora_projection --lora_mlp --lora_head"
    local global_batch_size=8
    local micro_batch_size=1
    local learning_rate=0.0001
    local optimizer="adamw"
    local weight_decay=0.0
    local patience=10
    local precision="bf16-true"

    # TRAIN
    local model_dir="$CHECKPOINTS_DIR/${model2checkpoint[$model]}"
    local log_dir="$train_dir/logs"
    local output_checkpoint_dir="$train_dir/checkpoint"
    if [ ! -f $train_dir/train_args.yaml ]; then
        
            # echo "Missing $train_dir/train_args.yaml"
         if [ $early_stopping = true ] && [ $train_list = "0.0-1.0" ]; then
            local max_steps=$(python -c "import torch; print(torch.load('$train_dir/../../0.0-0.7/0.7-1.0/best.ckpt', weights_only=False)['step_count'],end='')")
        else
            local max_steps=-1
        fi

        mkdir -p $train_dir $log_dir $output_checkpoint_dir
        for file in config.json generation_config.json model_config.yaml tokenizer.json tokenizer.model tokenizer_config.json; do
            if [ -f $model_dir/$file ]; then
                cp $model_dir/$file $output_checkpoint_dir
            fi
        done
        ln -sf $(readlink -f $model_dir/lit_model.pth) $output_checkpoint_dir/lit_model.pth
        python -m llmcal.scripts.train_lora \
            --base_checkpoint_dir $model_dir \
            --data_paths outputs/prompts/$model/$dataset/all.jsonl  \
            --train_lists lists/$dataset/size=$size/seed=$num_seed/$train_list.txt \
            --val_lists lists/$dataset/size=$size/seed=$num_seed/$val_list.txt \
            --output_dir $train_dir \
            --output_checkpoint_dir $output_checkpoint_dir \
            --log_dir $log_dir \
            --precision $precision \
            --devices 1 \
            --num_nodes 1 \
            --global_batch_size $global_batch_size \
            --micro_batch_size $micro_batch_size \
            --val_check_interval $val_check_interval \
            --learning_rate $learning_rate \
            --optimizer $optimizer \
            --weight_decay $weight_decay \
            --loss $loss \
            --patience $patience \
            --max_steps $max_steps \
            --seed $seed \
            $lora_args
    fi

    # PREDICT ON TEST
    local output_dir="$train_dir/test=$dataset/list=$test_list"
    if [ ! -f $output_dir/logits.csv ]; then
        mkdir -p $output_dir
        python -m llmcal.scripts.run_posteriors \
            --base_checkpoint_dir $model_dir \
            --checkpoint_dir $output_checkpoint_dir \
            --peft "lora" \
            --data_path outputs/prompts/$model/$dataset/all.jsonl \
            --output_dir $output_dir \
            --prediction_lists lists/$dataset/$test_list.txt \
            --precision $precision \
            --devices 1 \
            --num_nodes 1 \
            --batch_size 1 \
            --max_seq_length $max_seq_length \
            $lora_args
    fi
}


train_and_run_calibration() {
    local method=$1
    local num_seed=$2
    local train_dir=$3
    local pred_dir=$4
    local output_dir=$5
    
    if [[ "$method" == "dp_calibration" || "$method" == "temp_scaling" || "$method" == "vector_scaling" || "$method" == "bias_shift" ]]; then
        local script_name="llmcal.scripts.affine_calibration"
        local args_extra="--learning_rate 1e-3 --tolerance 1e-5 --max_ls 40"
    elif [[ "$method" == adats_* ]]; then
        local script_name="llmcal.scripts.adats_cal"
        local args_extra=""
    elif [[ "$method" == "dirichlet_fixed_diag" || "$method" == "dirichlet_full_l2" ]]; then
        local script_name="llmcal.scripts.dirichlet_cal"
        local args_extra=""
    else
        echo "Unknown method: $method"
        exit 1
    fi

    local seed=$((base_seed + num_seed))
    if [ ! -f $output_dir/logits.csv ]; then
        mkdir -p $output_dir
        python -m $script_name \
            --output_dir $output_dir \
            --log_dir "$output_dir/../../logs" \
            --checkpoint_dir "$output_dir/../../" \
            --train_logits "$train_dir/logits.csv" \
            --train_labels "$train_dir/labels.csv" \
            --predict_logits "$pred_dir/logits.csv" \
            --predict_labels "$pred_dir/labels.csv" \
            --method $method \
            $args_extra
    fi
}

run_calibration() {
    local method=$1
    local checkpoint_path=$2
    local predict_dir=$3
    local output_dir=$4


    if [[ "$method" == "dp_calibration" || "$method" == "temp_scaling" || "$method" == "vector_scaling" || "$method" == "bias_shift" ]]; then
        local script_name="llmcal.scripts.affine_prediction"
    elif [[ "$method" == adats_* ]]; then
        local script_name="llmcal.scripts.adats_pred"
    elif [[ "$method" == "dirichlet_fixed_diag" || "$method" == "dirichlet_full_l2" ]]; then
        local script_name="llmcal.scripts.dirichlet_pred"
    else
        echo "Unknown method: $method"
        exit 1
    fi

    if [ ! -f $output_dir/logits.csv ]; then
        mkdir -p $output_dir
        python -m $script_name \
            --checkpoint_path $checkpoint_path \
            --method $method \
            --predict_logits $predict_dir/logits.csv \
            --predict_labels $predict_dir/labels.csv \
            --output_dir $output_dir
    fi
}



declare -A calmethod2short=(
    ["dp_calibration"]="dpcal"
    ["temp_scaling"]="tempscaling"
    ["vector_scaling"]="vectorscaling"
    ["bias_shift"]="biasshift"
    ["adats_z_16"]="adatsz16"
    ["dirichlet_fixed_diag"]="dirichletfixeddiag"
    ["dirichlet_full_l2"]="dirichletfulll2"
)


# 1: model
# 2: sizes
# 3: val_check_interval
run_lora_vs_samples() {
    local model=$1
    local val_check_interval=$2
    for size in ${FACTORS[@]}; do
        for dataset in "${DATASETS[@]}"; do
            local test_list="test_${dataset2testsize[$dataset]}"
            local num_seeds=${dataset2nseeds[$dataset]}
            for num_seed in $(seq 0 $(($num_seeds - 1))); do

                # Train lora-ans without regularization + calibration
                train_list1="0.0-0.7"
                val_list1="0.0-0.3"
                cal_list="0.7-1.0"
                train_dir1="outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list1/$val_list1"
                mkdir -p $train_dir1
                run_lora $model $dataset $size ans $num_seed $val_check_interval $train_dir1 false $train_list1 $val_list1 $cal_list
                for calmethod in "${!calmethod2short[@]}"; do
                    train_and_run_calibration "$calmethod" $num_seed \
                        "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list1/$val_list1/test=$dataset/list=$cal_list" \
                        "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list1/$val_list1/test=$dataset/list=$test_list" \
                        "outputs/lora_plus_${calmethod2short[$calmethod]}/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list1/$val_list1/$cal_list/test=$dataset/list=$test_list"
                done
                train_list2="0.0-1.0"
                val_list2="0.7-1.0"
                train_dir2="outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list2/$val_list2"
                mkdir -p $train_dir2
                run_lora $model $dataset $size ans $num_seed $val_check_interval $train_dir2 false $train_list2 $val_list2 $test_list
                for calmethod in "${!calmethod2short[@]}"; do
                    run_calibration "$calmethod" \
                        "outputs/lora_plus_${calmethod2short[$calmethod]}/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list1/$val_list1/$cal_list/state.ckpt" \
                        "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list2/$val_list2/test=$dataset/list=$test_list" \
                        "outputs/lora_plus_${calmethod2short[$calmethod]}/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list2/$val_list2/$cal_list/test=$dataset/list=$test_list"
                done
                
                # Train lora-ans LS (alpha=0.1) + calibration
                train_list1="0.0-0.7"
                val_list1="0.0-0.3"
                cal_list="0.7-1.0"
                train_dir1="outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_ls-0.1/$train_list1/$val_list1"
                mkdir -p $train_dir1
                run_lora $model $dataset $size ans-ls_0.1 $num_seed $val_check_interval $train_dir1 false $train_list1 $val_list1 $cal_list
                for calmethod in "${!calmethod2short[@]}"; do
                    train_and_run_calibration "$calmethod" $num_seed \
                        "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_ls-0.1/$train_list1/$val_list1/test=$dataset/list=$cal_list" \
                        "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_ls-0.1/$train_list1/$val_list1/test=$dataset/list=$test_list" \
                        "outputs/lora_plus_${calmethod2short[$calmethod]}/$model/$dataset/size=$size/seed=$num_seed/lora_ans_ls-0.1/$train_list1/$val_list1/$cal_list/test=$dataset/list=$test_list"
                done
                train_list2="0.0-1.0"
                val_list2="0.7-1.0"
                train_dir2="outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_ls-0.1/$train_list2/$val_list2"
                mkdir -p $train_dir2
                run_lora $model $dataset $size ans-ls_0.1 $num_seed $val_check_interval $train_dir2 false $train_list2 $val_list2 $test_list
                for calmethod in "${!calmethod2short[@]}"; do
                    run_calibration "$calmethod" \
                        "outputs/lora_plus_${calmethod2short[$calmethod]}/$model/$dataset/size=$size/seed=$num_seed/lora_ans_ls-0.1/$train_list1/$val_list1/$cal_list/state.ckpt" \
                        "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_ls-0.1/$train_list2/$val_list2/test=$dataset/list=$test_list" \
                        "outputs/lora_plus_${calmethod2short[$calmethod]}/$model/$dataset/size=$size/seed=$num_seed/lora_ans_ls-0.1/$train_list2/$val_list2/$cal_list/test=$dataset/list=$test_list"
                done


            done
        done
    done
}

run_lora_vs_samples $model 16

