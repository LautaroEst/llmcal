#!/bin/bash

### Finetuning
python -m llmcal agnews_4_926 encoder_agnews roberta_base full_ft no_calibration \
    --batch_size 32 \
    --accumulate_grad_batches 1 \
    --learning_rate 1e-5 \
    --max_epochs 30 \
    --val_check_interval 1 \
    --max_steps -1

python -m llmcal agnews_16_564 encoder_agnews roberta_base full_ft no_calibration \
    --batch_size 32 \
    --accumulate_grad_batches 1 \
    --learning_rate 1e-5 \
    --max_epochs 30 \
    --val_check_interval 1 \
    --max_steps -1

python -m llmcal agnews_256_821 encoder_agnews roberta_base full_ft no_calibration \
    --batch_size 32 \
    --accumulate_grad_batches 1 \
    --learning_rate 1e-5 \
    --max_epochs 30 \
    --val_check_interval 1 \
    --max_steps -1
