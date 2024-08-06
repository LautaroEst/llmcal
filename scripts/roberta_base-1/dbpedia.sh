#!/bin/bash

### Finetuning
python -m llmcal dbpedia_2_927 encoder_dbpedia roberta_base full_ft no_calibration \
    --batch_size 32 \
    --accumulate_grad_batches 1 \
    --learning_rate 1e-5 \
    --max_epochs 80 \
    --val_check_interval 1 \
    --max_steps -1

# python -m llmcal dbpedia_4_295 encoder_dbpedia roberta_base full_ft no_calibration \
#     --batch_size 32 \
#     --accumulate_grad_batches 1 \
#     --learning_rate 1e-5 \
#     --max_epochs 30 \
#     --val_check_interval 1 \
#     --max_steps -1

# python -m llmcal dbpedia_16_738 encoder_dbpedia roberta_base full_ft no_calibration \
#     --batch_size 32 \
#     --accumulate_grad_batches 1 \
#     --learning_rate 1e-5 \
#     --max_epochs 30 \
#     --val_check_interval 1 \
#     --max_steps -1

# python -m llmcal dbpedia_256_493 encoder_dbpedia roberta_base full_ft no_calibration \
#     --batch_size 32 \
#     --accumulate_grad_batches 1 \
#     --learning_rate 1e-5 \
#     --max_epochs 2 \
#     --val_check_interval 1 \
#     --max_steps -1
