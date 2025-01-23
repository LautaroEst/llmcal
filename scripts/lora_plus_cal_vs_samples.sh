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
# 11: pred_list
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
    local pred_list=${11}
    local lora_args="--lora_r=8 --lora_alpha=16 --lora_dropout=0.05 --lora_query --lora_key --lora_value --lora_projection --lora_mlp --lora_head"
    local global_batch_size=8
    local micro_batch_size=1
    local learning_rate=0.0001
    local optimizer="adamw"
    local weight_decay=0.0
    local patience=10
    local precision="bf16-true"

    if [ $early_stopping = true ] && [ $train_list = "0.0-1.0" ]; then
        local max_steps=$(python -c "import torch; print(torch.load('$train_dir/../../0.0-0.7/0.7-1.0/best.ckpt', weights_only=False)['step_count'],end='')")
    else
        local max_steps=-1
    fi

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

    # PREDICT
    local output_dir="$train_dir/test=$dataset/list=$pred_list"
    if [ ! -f $output_dir/logits.csv ]; then
        mkdir -p $output_dir
        python -m llmcal.scripts.run_posteriors \
            --base_checkpoint_dir $model_dir \
            --checkpoint_dir $output_checkpoint_dir \
            --peft "lora" \
            --data_path outputs/prompts/$model/$dataset/all.jsonl \
            --output_dir $output_dir \
            --prediction_lists lists/$dataset/size=$size/seed=$num_seed/$pred_list.txt \
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
    
    local seed=$((base_seed + num_seed))
    if [ ! -f $output_dir/logits.csv ]; then
        mkdir -p $output_dir
        python -m llmcal.scripts.affine_calibration \
            --output_dir $output_dir \
            --log_dir "$output_dir/../../logs" \
            --checkpoint_dir "$output_dir/../../" \
            --train_logits "$train_dir/logits.csv" \
            --train_labels "$train_dir/labels.csv" \
            --predict_logits "$pred_dir/logits.csv" \
            --predict_labels "$pred_dir/labels.csv" \
            --method $method \
            --learning_rate 1e-3 \
            --tolerance 1e-5 \
            --max_ls 40
    fi
}

run_calibration() {
    local method=$1
    local checkpoint_path=$2
    local predict_dir=$3
    local output_dir=$4
    if [ ! -f $output_dir/logits.csv ]; then
        mkdir -p $output_dir
        python -m llmcal.scripts.affine_prediction \
            --checkpoint_path $checkpoint_path \
            --method $method \
            --predict_logits $predict_dir/logits.csv \
            --predict_labels $predict_dir/labels.csv \
            --output_dir $output_dir
    fi
}




# 1: model
# 2: sizes
# 3: val_check_interval
run_lora_vs_samples() {
    local model=$1
    local val_check_interval=$2
    for size in ${FACTORS[@]}; do
        for dataset in "${DATASETS[@]}"; do
            local test_list="test_${dataset2testsize[$dataset]}"
            for num_seed in $(seq 0 $(($num_seeds - 1))); do

                # Train lora-ans without early stopping on 70% of the data and calibrate on 30% of the data
                train_list="0.0-0.7"
                val_list="0.0-0.3"
                pred_list="0.7-1.0"
                train_dir="outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list"
                run_lora $model $dataset $size ans $num_seed $val_check_interval $train_dir false $train_list $val_list $pred_list
                train_and_run_calibration "dp_calibration" $num_seed \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/test=$dataset/list=$pred_list" \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/test=$dataset/list=$test_list" \
                    "outputs/lora_plus_dpcal/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/$pred_list/test=$dataset/list=$test_list"
                train_and_run_calibration "temp_scaling" $num_seed \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/test=$dataset/list=$pred_list" \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/test=$dataset/list=$test_list" \
                    "outputs/lora_plus_tempscaling/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/$pred_list/test=$dataset/list=$test_list"

                # Train lora-ans without early stopping on 100% of the data and calibrate using the above calibrated model
                train_list="0.0-1.0"
                val_list="0.7-1.0"
                pred_list="0.7-1.0"
                train_dir="outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list"
                mkdir -p $train_dir
                run_lora $model $dataset $size ans $num_seed $val_check_interval $train_dir false $train_list $val_list $test_list
                run_calibration "dp_calibration" \
                    "outputs/lora_plus_dpcal/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/0.0-0.7/0.0-0.3/$pred_list/state.ckpt" \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/test=$dataset/list=$test_list" \
                    "outputs/lora_plus_dpcal/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/$pred_list/test=$dataset/list=$test_list"
                run_calibration "temp_scaling" \
                    "outputs/lora_plus_tempscaling/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/0.0-0.7/0.0-0.3/$pred_list/state.ckpt" \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/test=$dataset/list=$test_list" \
                    "outputs/lora_plus_tempscaling/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/$pred_list/test=$dataset/list=$test_list"

                # Train lora-ans without early stopping on 100% of the data + naive calibration
                train_list="0.0-1.0"
                val_list="0.7-1.0"
                pred_list="0.0-1.0"
                train_dir="outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list"
                mkdir -p $train_dir
                run_lora $model $dataset $size ans $num_seed $val_check_interval $train_dir false $train_list $val_list $pred_list
                train_and_run_calibration "dp_calibration" $num_seed \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/test=$dataset/list=$pred_list" \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/test=$dataset/list=$test_list" \
                    "outputs/lora_plus_dpcal_naive/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/$pred_list/test=$dataset/list=$test_list"
                train_and_run_calibration "temp_scaling" $num_seed \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/test=$dataset/list=$pred_list" \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/test=$dataset/list=$test_list" \
                    "outputs/lora_plus_tempscaling_naive/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/$pred_list/test=$dataset/list=$test_list"

                # Train lora-ans without early stopping on 100% of the data + calibration train on test
                train_list="0.0-1.0"
                val_list="0.7-1.0"
                pred_list="0.7-1.0"
                train_dir="outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list"
                mkdir -p $train_dir
                run_lora $model $dataset $size ans $num_seed $val_check_interval $train_dir false $train_list $val_list $test_list
                train_and_run_calibration "dp_calibration" $num_seed \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/test=$dataset/list=$test_list" \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/test=$dataset/list=$test_list" \
                    "outputs/lora_plus_dpcal_trainontest/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/$test_list/test=$dataset/list=$test_list"
                train_and_run_calibration "temp_scaling" $num_seed \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/test=$dataset/list=$test_list" \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/test=$dataset/list=$test_list" \
                    "outputs/lora_plus_tempscaling_trainontest/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/$test_list/test=$dataset/list=$test_list"

                # Train lora-ans with early stopping on 70% of the data and calibrate on 30% of the data
                train_list="0.0-0.7"
                val_list="0.7-1.0"
                pred_list="0.7-1.0"
                train_dir="outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans/$train_list/$val_list"
                mkdir -p $train_dir
                run_lora $model $dataset $size ans $num_seed $val_check_interval $train_dir true $train_list $val_list $pred_list
                train_and_run_calibration "dp_calibration" $num_seed \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans/$train_list/$val_list/test=$dataset/list=$pred_list" \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans/$train_list/$val_list/test=$dataset/list=$test_list" \
                    "outputs/lora_plus_dpcal/$model/$dataset/size=$size/seed=$num_seed/lora_ans/$train_list/$val_list/$pred_list/test=$dataset/list=$test_list"
                train_and_run_calibration "temp_scaling" $num_seed \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans/$train_list/$val_list/test=$dataset/list=$pred_list" \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans/$train_list/$val_list/test=$dataset/list=$test_list" \
                    "outputs/lora_plus_tempscaling/$model/$dataset/size=$size/seed=$num_seed/lora_ans/$train_list/$val_list/$pred_list/test=$dataset/list=$test_list"

                # Train lora-ans with early stopping on 100% of the data and calibrate using the above calibrated model
                train_list="0.0-1.0"
                val_list="0.7-1.0"
                pred_list="0.7-1.0"
                train_dir="outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans/$train_list/$val_list"
                mkdir -p $train_dir
                run_lora $model $dataset $size ans $num_seed $val_check_interval $train_dir true $train_list $val_list $test_list
                run_calibration "dp_calibration" \
                    "outputs/lora_plus_dpcal/$model/$dataset/size=$size/seed=$num_seed/lora_ans/0.0-0.7/0.7-1.0/$pred_list/state.ckpt" \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans/$train_list/$val_list/test=$dataset/list=$test_list" \
                    "outputs/lora_plus_dpcal/$model/$dataset/size=$size/seed=$num_seed/lora_ans/$train_list/$val_list/$pred_list/test=$dataset/list=$test_list"
                run_calibration "temp_scaling" \
                    "outputs/lora_plus_tempscaling/$model/$dataset/size=$size/seed=$num_seed/lora_ans/0.0-0.7/0.7-1.0/$pred_list/state.ckpt" \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans/$train_list/$val_list/test=$dataset/list=$test_list" \
                    "outputs/lora_plus_tempscaling/$model/$dataset/size=$size/seed=$num_seed/lora_ans/$train_list/$val_list/$pred_list/test=$dataset/list=$test_list"
                
                # Train lora-ans with early stopping on 100% of the data + calibration on test set
                train_list="0.0-1.0"
                val_list="0.7-1.0"
                train_dir="outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans/$train_list/$val_list"
                mkdir -p $train_dir
                run_lora $model $dataset $size ans $num_seed $val_check_interval $train_dir true $train_list $val_list $test_list
                train_and_run_calibration "dp_calibration" $num_seed \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans/$train_list/$val_list/test=$dataset/list=$test_list" \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans/$train_list/$val_list/test=$dataset/list=$test_list" \
                    "outputs/lora_plus_dpcal_trainontest/$model/$dataset/size=$size/seed=$num_seed/lora_ans/$train_list/$val_list/$test_list/test=$dataset/list=$test_list"
                train_and_run_calibration "temp_scaling" $num_seed \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans/$train_list/$val_list/test=$dataset/list=$test_list" \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans/$train_list/$val_list/test=$dataset/list=$test_list" \
                    "outputs/lora_plus_tempscaling_trainontest/$model/$dataset/size=$size/seed=$num_seed/lora_ans/$train_list/$val_list/$test_list/test=$dataset/list=$test_list"
            done
        done
    done
}

run_lora_vs_samples $model 16
