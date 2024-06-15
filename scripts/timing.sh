#!/bin/bash

timit () {
    dataset=$1
    seed=$2
    lora_config=$3
    python -m llmcal ${dataset}_${seed} basic_${dataset}_0-shot_litgpt lm_tinyllama no_adaptation_bf16 affine_scalar --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30 --timing
    python -m llmcal ${dataset}_${seed} basic_${dataset}_0-shot_litgpt lm_tinyllama $lora_config no_calibration --accelerator "gpu" --timing
}

# timit sst2 8_639 lora_10samples
# timit sst2 16_738 lora_20samples
# timit sst2 256_493 lora_500samples

# timit agnews 4_295 lora_10samples
# timit agnews 8_639 lora_20samples
# timit agnews 256_493 lora_1000samples

timit dbpedia 2_927 lora_20samples
timit dbpedia 4_295 lora_60samples
timit dbpedia 256_493 lora_3500samples

timit 20newsgroups 2_927 lora_40samples
timit 20newsgroups 4_295 lora_60samples
timit 20newsgroups 256_493 lora_3500samples

timit banking77 4_295 lora_200samples
timit banking77 16_738 lora_1000samples
timit banking77 64_893 lora_5000samples