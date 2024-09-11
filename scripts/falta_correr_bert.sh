export CUDA_VISIBLE_DEVICES=1
bash scripts/roberta_base-1/20newsgroups.sh
bash scripts/roberta_base-2/20newsgroups.sh
bash scripts/roberta_base-3/20newsgroups.sh
bash scripts/roberta_base-4/20newsgroups.sh
bash scripts/roberta_base-5/20newsgroups.sh

python -m llmcal dbpedia_2_9722 encoder_dbpedia roberta_base full_ft no_calibration \
    --batch_size 32 \
    --accumulate_grad_batches 1 \
    --learning_rate 1e-5 \
    --max_epochs 80 \
    --val_check_interval 1 \
    --max_steps -1

python -m llmcal dbpedia_128_129 encoder_dbpedia roberta_base full_ft no_calibration \
    --batch_size 32 \
    --accumulate_grad_batches 1 \
    --learning_rate 1e-5 \
    --max_epochs 80 \
    --val_check_interval 1 \
    --max_steps -1

python -m llmcal dbpedia_128_131 encoder_dbpedia roberta_base full_ft no_calibration \
    --batch_size 32 \
    --accumulate_grad_batches 1 \
    --learning_rate 1e-5 \
    --max_epochs 80 \
    --val_check_interval 1 \
    --max_steps -1