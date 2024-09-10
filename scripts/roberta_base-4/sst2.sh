# #!/bin/bash

### Finetuning
python -m llmcal sst2_8_6391 encoder_sst2 roberta_base full_ft no_calibration \
    --batch_size 32 \
    --accumulate_grad_batches 1 \
    --learning_rate 1e-5 \
    --max_epochs 30 \
    --val_check_interval 1 \
    --max_steps -1

python -m llmcal sst2_32_15641 encoder_sst2 roberta_base full_ft no_calibration \
    --batch_size 32 \
    --accumulate_grad_batches 1 \
    --learning_rate 1e-5 \
    --max_epochs 30 \
    --val_check_interval 1 \
    --max_steps -1

python -m llmcal sst2_512_890 encoder_sst2 roberta_base full_ft no_calibration \
    --batch_size 32 \
    --accumulate_grad_batches 1 \
    --learning_rate 1e-5 \
    --max_epochs 30 \
    --val_check_interval 1 \
    --max_steps -1
