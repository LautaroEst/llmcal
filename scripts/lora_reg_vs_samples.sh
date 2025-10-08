#!/bin/bash -ex

source ./scripts/env.sh

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
run_lora_reg() {
    local model=$1
    local dataset=$2
    local size=$3
    local loss=$4
    local num_seed=$5
    local seed=$((base_seed + num_seed))
    local val_check_interval=$6
    local train_dir=$7
    local train_list=$8
    local val_list=${9}
    local test_list=${10}
    local lora_args="--lora_r=8 --lora_alpha=16 --lora_dropout=0.05 --lora_query --lora_key --lora_value --lora_projection --lora_mlp --lora_head"
    local global_batch_size=8
    local micro_batch_size=1
    local learning_rate=0.0001
    local optimizer="adamw"
    local weight_decay=0.0
    local patience=10
    local precision="bf16-true"
    local max_steps=-1

    # TRAIN
    local model_dir="$CHECKPOINTS_DIR/${model2checkpoint[$model]}"
    local log_dir="$train_dir/logs"
    local output_checkpoint_dir="$train_dir/checkpoint"
    if [ ! -f $train_dir/train_args.yaml ]; then
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


# 1: model
# 2: sizes
# 3: val_check_interval
run_lora_vs_samples() {
    local model=$1
    local val_check_interval=$2
    for size in ${FACTORS[@]}; do
        # local num_seeds=${dataset2nseeds[$dataset]}
        local num_seeds=3
        # for num_seed in $(seq 0 $(($num_seeds - 1))); do
        for num_seed in 2 ; do
            for dataset in "${DATASETS[@]}"; do
                local test_list="test_${dataset2testsize[$dataset]}"

                # Train lora-ans with L2 regularization (lambda=0.1)
                train_list="0.0-1.0"
                val_list="0.7-1.0"
                train_dir="outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_l2-0.1/$train_list/$val_list"
                mkdir -p $train_dir
                run_lora_reg $model $dataset $size ans-l2_0.1 $num_seed $val_check_interval $train_dir $train_list $val_list $test_list

                # Train lora-ans with label smoothing regularization (lambda=0.1)
                train_list="0.0-1.0"
                val_list="0.7-1.0"
                train_dir="outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_ls-0.1/$train_list/$val_list"
                mkdir -p $train_dir
                run_lora_reg $model $dataset $size ans-ls_0.1 $num_seed $val_check_interval $train_dir $train_list $val_list $test_list

                # Train lora-ans with L2 regularization (lambda=0.01)
                train_list="0.0-1.0"
                val_list="0.7-1.0"
                train_dir="outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_l2-0.01/$train_list/$val_list"
                mkdir -p $train_dir
                run_lora_reg $model $dataset $size ans-l2_0.01 $num_seed $val_check_interval $train_dir $train_list $val_list $test_list

                # Train lora-ans with label smoothing regularization (lambda=0.01)
                train_list="0.0-1.0"
                val_list="0.7-1.0"
                train_dir="outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_ls-0.01/$train_list/$val_list"
                mkdir -p $train_dir
                run_lora_reg $model $dataset $size ans-ls_0.01 $num_seed $val_check_interval $train_dir $train_list $val_list $test_list
            done
        done
    done
}

run_lora_vs_samples $model 16

