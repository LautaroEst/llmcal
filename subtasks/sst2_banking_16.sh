#!/bin/bash -ex

source ./subtasks/base.sh

size=16
num_seed=0

for dataset in sst2 banking77; do
    # Train lora-ans with label smoothing regularization (lambda=0.5)
    train_list="0.0-1.0"
    val_list="0.7-1.0"
    train_dir="outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_ls-0.5/$train_list/$val_list"
    test_list="test_${dataset2testsize[$dataset]}"
    mkdir -p $train_dir
    run_lora_reg $model $dataset $size ans-ls_0.5 $num_seed $val_check_interval $train_dir $train_list $val_list $test_list


    # Train lora-ans with label smoothing regularization (lambda=0.5)
    train_list="0.0-0.7"
    val_list="0.7-1.0"
    train_dir="outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_ls-0.5/$train_list/$val_list"
    test_list="test_${dataset2testsize[$dataset]}"
    mkdir -p $train_dir
    run_lora_reg $model $dataset $size ans-ls_0.5 $num_seed $val_check_interval $train_dir $train_list $val_list $test_list
done