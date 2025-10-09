#!/bin/bash -ex

source ./subtasks/base.sh

size=16


for num_seed in 0 1 2; do
    for dataset in agnews dbpedia 20newsgroups; do
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
        train_dir="outputs/finetune_lora/$model/$dataset/size=$size/seed=$num_seed/lora_ans_ls-0.1/$train_list/$val_list"
        test_list="test_${dataset2testsize[$dataset]}"
        mkdir -p $train_dir
        run_lora_reg $model $dataset $size ans-ls_0.1 $num_seed $val_check_interval $train_dir $train_list $val_list $test_list

        output_dir="$train_dir/test=$dataset/list=$val_list"
        model_dir="$CHECKPOINTS_DIR/${model2checkpoint[$model]}"
        output_checkpoint_dir="$train_dir/checkpoint"
        lora_args="--lora_r=8 --lora_alpha=16 --lora_dropout=0.05 --lora_query --lora_key --lora_value --lora_projection --lora_mlp --lora_head"
        if [ ! -f $output_dir/logits.csv ]; then
            mkdir -p $output_dir
            python -m llmcal.scripts.run_posteriors \
                --base_checkpoint_dir $model_dir \
                --checkpoint_dir $output_checkpoint_dir \
                --peft "lora" \
                --data_path outputs/prompts/$model/$dataset/all.jsonl \
                --output_dir $output_dir \
                --prediction_lists lists/$dataset/size=$size/seed=$num_seed/$val_list.txt \
                --precision "bf16-true" \
                --devices 1 \
                --num_nodes 1 \
                --batch_size 1 \
                --max_seq_length $max_seq_length \
                $lora_args
        fi
    done
done