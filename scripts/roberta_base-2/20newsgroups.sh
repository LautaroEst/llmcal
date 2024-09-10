#!/bin/bash


### Finetuning
python -m llmcal 20newsgroups_2_435 encoder_20newsgroups roberta_base full_ft no_calibration \
    --batch_size 32 \
    --accumulate_grad_batches 1 \
    --learning_rate 1e-5 \
    --max_epochs 30 \
    --val_check_interval 1 \
    --max_steps -1

python -m llmcal 20newsgroups_8_364 encoder_20newsgroups roberta_base full_ft no_calibration \
    --batch_size 32 \
    --accumulate_grad_batches 1 \
    --learning_rate 1e-5 \
    --max_epochs 30 \
    --val_check_interval 1 \
    --max_steps -1

python -m llmcal 20newsgroups_128_131 encoder_20newsgroups roberta_base full_ft no_calibration \
    --batch_size 32 \
    --accumulate_grad_batches 1 \
    --learning_rate 1e-5 \
    --max_epochs 30 \
    --val_check_interval 1 \
    --max_steps -1
