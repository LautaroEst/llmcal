#!/bin/bash

dataset=sst2
for suffix in 8_639 8_923 8_932 8_6391 8_9322; do
    cache=experiments/${dataset}_${suffix}/basic_${dataset}_0-shot_litgpt/lm_tinyllama/lora_xval_10samples/.cache/predictions
    mkdir -p $cache
    ln -sf ../../../lora_10samples/.cache/predictions/test $cache
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_10samples no_calibration --accelerator "gpu" --batch_size 1 
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_10samples affine_scalar --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_10samples temp_scaling --accelerator "gpu" --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_10samples bias_only --accelerator "gpu" --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
done

dataset=agnews
for suffix in 8_639 8_923 8_932 8_6391 8_9322; do
    cache=experiments/${dataset}_${suffix}/basic_${dataset}_0-shot_litgpt/lm_tinyllama/lora_xval_20samples/.cache/predictions
    mkdir -p $cache
    ln -sf ../../../lora_20samples/.cache/predictions/test $cache
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_20samples no_calibration --accelerator "gpu" --batch_size 1 
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_20samples affine_scalar --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_20samples temp_scaling --accelerator "gpu" --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_20samples bias_only --accelerator "gpu" --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
done

dataset=dbpedia
for suffix in 2_435 2_927 2_972 2_4351 2_9722; do
    cache=experiments/${dataset}_${suffix}/basic_${dataset}_0-shot_litgpt/lm_tinyllama/lora_xval_20samples/.cache/predictions
    mkdir -p $cache
    ln -sf ../../../lora_20samples/.cache/predictions/test $cache
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_20samples no_calibration --accelerator "gpu" --batch_size 1 
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_20samples affine_scalar --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_20samples temp_scaling --accelerator "gpu" --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_20samples bias_only --accelerator "gpu" --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
done

dataset=20newsgroups # TODO: Es con 40samples!!
for suffix in 2_435 2_927 2_972 2_9722; do
    cache=experiments/${dataset}_${suffix}/basic_${dataset}_0-shot_litgpt/lm_tinyllama/lora_xval_20samples/.cache/predictions
    mkdir -p $cache
    ln -sf ../../../lora_20samples/.cache/predictions/test $cache
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_20samples no_calibration --accelerator "gpu" --batch_size 1 
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_20samples affine_scalar --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_20samples temp_scaling --accelerator "gpu" --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_20samples bias_only --accelerator "gpu" --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
done

dataset=sst2
for suffix in 16_564 16_738 16_783 16_5641 16_7832; do
    cache=experiments/${dataset}_${suffix}/basic_${dataset}_0-shot_litgpt/lm_tinyllama/lora_xval_20samples/.cache/predictions
    mkdir -p $cache
    ln -sf ../../../lora_20samples/.cache/predictions/test $cache
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_20samples no_calibration --accelerator "gpu" --batch_size 1 
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_20samples affine_scalar --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_20samples temp_scaling --accelerator "gpu" --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_20samples bias_only --accelerator "gpu" --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
done

dataset=agnews
for suffix in 16_564 16_738 16_783 16_5641 16_7832; do
    cache=experiments/${dataset}_${suffix}/basic_${dataset}_0-shot_litgpt/lm_tinyllama/lora_xval_60samples/.cache/predictions
    mkdir -p $cache
    ln -sf ../../../lora_60samples/.cache/predictions/test $cache
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_60samples no_calibration --accelerator "gpu" --batch_size 1 
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_60samples affine_scalar --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_60samples temp_scaling --accelerator "gpu" --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_60samples bias_only --accelerator "gpu" --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
done

dataset=dbpedia
for suffix in 4_295 4_926 4_962 4_2951 4_9622; do
    cache=experiments/${dataset}_${suffix}/basic_${dataset}_0-shot_litgpt/lm_tinyllama/lora_xval_60samples/.cache/predictions
    mkdir -p $cache
    ln -sf ../../../lora_60samples/.cache/predictions/test $cache
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_60samples no_calibration --accelerator "gpu" --batch_size 1 
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_60samples affine_scalar --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_60samples temp_scaling --accelerator "gpu" --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_60samples bias_only --accelerator "gpu" --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
done

dataset=20newsgroups
for suffix in 4_295 4_926 4_962 4_9622; do
    cache=experiments/${dataset}_${suffix}/basic_${dataset}_0-shot_litgpt/lm_tinyllama/lora_xval_60samples/.cache/predictions
    mkdir -p $cache
    ln -sf ../../../lora_60samples/.cache/predictions/test $cache
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_60samples no_calibration --accelerator "gpu" --batch_size 1 
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_60samples affine_scalar --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_60samples temp_scaling --accelerator "gpu" --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_60samples bias_only --accelerator "gpu" --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
done

dataset=sst2
for suffix in 256_493 256_812 256_821 256_4931 256_8212; do
    cache=experiments/${dataset}_${suffix}/basic_${dataset}_0-shot_litgpt/lm_tinyllama/lora_xval_500samples/.cache/predictions
    mkdir -p $cache
    ln -sf ../../../lora_500samples/.cache/predictions/test $cache
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_500samples no_calibration --accelerator "gpu" --batch_size 1 
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_500samples affine_scalar --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_500samples temp_scaling --accelerator "gpu" --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_500samples bias_only --accelerator "gpu" --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
done

dataset=agnews
for suffix in 256_493 256_812 256_821 256_4931 256_8212; do
    cache=experiments/${dataset}_${suffix}/basic_${dataset}_0-shot_litgpt/lm_tinyllama/lora_xval_1000samples/.cache/predictions
    mkdir -p $cache
    ln -sf ../../../lora_1000samples/.cache/predictions/test $cache
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_1000samples no_calibration --accelerator "gpu" --batch_size 1 
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_1000samples affine_scalar --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_1000samples temp_scaling --accelerator "gpu" --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_1000samples bias_only --accelerator "gpu" --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
done

dataset=dbpedia
for suffix in 256_493 256_812 256_821 256_4931 256_8212; do
    cache=experiments/${dataset}_${suffix}/basic_${dataset}_0-shot_litgpt/lm_tinyllama/lora_xval_3500samples/.cache/predictions
    mkdir -p $cache
    ln -sf ../../../lora_3500samples/.cache/predictions/test $cache
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_3500samples no_calibration --accelerator "gpu" --batch_size 1 
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_3500samples affine_scalar --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_3500samples temp_scaling --accelerator "gpu" --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_3500samples bias_only --accelerator "gpu" --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
done

dataset=20newsgroups
for suffix in 256_493 256_812 256_8212; do
    cache=experiments/${dataset}_${suffix}/basic_${dataset}_0-shot_litgpt/lm_tinyllama/lora_xval_3500samples/.cache/predictions
    mkdir -p $cache
    ln -sf ../../../lora_3500samples/.cache/predictions/test $cache
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_3500samples no_calibration --accelerator "gpu" --batch_size 1 
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_3500samples affine_scalar --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_3500samples temp_scaling --accelerator "gpu" --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_xval_3500samples bias_only --accelerator "gpu" --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
done