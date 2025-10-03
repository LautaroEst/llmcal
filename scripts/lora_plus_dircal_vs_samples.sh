#!/bin/bash -ex

source ./scripts/env.sh

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
            local num_seeds=${dataset2nseeds[$dataset]}
            for num_seed in $(seq 0 $(($num_seeds - 1))); do

                # Train lora-ans without early stopping on 70% of the data and calibrate on 30% of the data
                train_list="0.0-0.7"
                val_list="0.0-0.3"
                pred_list="0.7-1.0"
                train_dir="outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list"
                train_and_run_calibration "dirichlet_fixed_diag" $num_seed \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/test=$dataset/list=$pred_list" \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/test=$dataset/list=$test_list" \
                    "outputs/lora_plus_dpcal/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/$pred_list/test=$dataset/list=$test_list"
                
                # Train lora-ans without early stopping on 100% of the data and calibrate using the above calibrated model
                train_list="0.0-1.0"
                val_list="0.7-1.0"
                pred_list="0.7-1.0"
                train_dir="outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list"
                mkdir -p $train_dir
                run_calibration "dirichlet_fixed_diag" \
                    "outputs/lora_plus_dpcal/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/0.0-0.7/0.0-0.3/$pred_list/state.ckpt" \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/test=$dataset/list=$test_list" \
                    "outputs/lora_plus_dpcal/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/$pred_list/test=$dataset/list=$test_list"
                
                # Train lora-ans without early stopping on 100% of the data + naive calibration
                train_list="0.0-1.0"
                val_list="0.7-1.0"
                pred_list="0.0-1.0"
                train_dir="outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list"
                mkdir -p $train_dir
                train_and_run_calibration "dirichlet_fixed_diag" $num_seed \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/test=$dataset/list=$pred_list" \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/test=$dataset/list=$test_list" \
                    "outputs/lora_plus_dpcal_naive/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/$pred_list/test=$dataset/list=$test_list"
                
                # Train lora-ans without early stopping on 100% of the data + calibration train on test
                train_list="0.0-1.0"
                val_list="0.7-1.0"
                pred_list="0.7-1.0"
                train_dir="outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list"
                mkdir -p $train_dir
                train_and_run_calibration "dirichlet_fixed_diag" $num_seed \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/test=$dataset/list=$test_list" \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/test=$dataset/list=$test_list" \
                    "outputs/lora_plus_dpcal_trainontest/$model/$dataset/size=$size/seed=$num_seed/lora_ans_no_es/$train_list/$val_list/$test_list/test=$dataset/list=$test_list"
                
                # Train lora-ans with early stopping on 70% of the data and calibrate on 30% of the data
                train_list="0.0-0.7"
                val_list="0.7-1.0"
                pred_list="0.7-1.0"
                train_dir="outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans/$train_list/$val_list"
                mkdir -p $train_dir
                train_and_run_calibration "dirichlet_fixed_diag" $num_seed \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans/$train_list/$val_list/test=$dataset/list=$pred_list" \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans/$train_list/$val_list/test=$dataset/list=$test_list" \
                    "outputs/lora_plus_dpcal/$model/$dataset/size=$size/seed=$num_seed/lora_ans/$train_list/$val_list/$pred_list/test=$dataset/list=$test_list"
                
                # Train lora-ans with early stopping on 100% of the data and calibrate using the above calibrated model
                train_list="0.0-1.0"
                val_list="0.7-1.0"
                pred_list="0.7-1.0"
                train_dir="outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans/$train_list/$val_list"
                mkdir -p $train_dir
                run_calibration "dirichlet_fixed_diag" \
                    "outputs/lora_plus_dpcal/$model/$dataset/size=$size/seed=$num_seed/lora_ans/0.0-0.7/0.7-1.0/$pred_list/state.ckpt" \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans/$train_list/$val_list/test=$dataset/list=$test_list" \
                    "outputs/lora_plus_dpcal/$model/$dataset/size=$size/seed=$num_seed/lora_ans/$train_list/$val_list/$pred_list/test=$dataset/list=$test_list"
                
                # Train lora-ans with early stopping on 100% of the data + calibration on test set
                train_list="0.0-1.0"
                val_list="0.7-1.0"
                train_dir="outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans/$train_list/$val_list"
                mkdir -p $train_dir
                train_and_run_calibration "dirichlet_fixed_diag" $num_seed \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans/$train_list/$val_list/test=$dataset/list=$test_list" \
                    "outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans/$train_list/$val_list/test=$dataset/list=$test_list" \
                    "outputs/lora_plus_dpcal_trainontest/$model/$dataset/size=$size/seed=$num_seed/lora_ans/$train_list/$val_list/$test_list/test=$dataset/list=$test_list"
            done
        done
    done
}

run_lora_vs_samples $model 16
