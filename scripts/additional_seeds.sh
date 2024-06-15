#!/bin/bash

dataset=sst2
for seed in 8_901 8_802 8_703 8_604 8_505; do
    mkdir -p experiments/${dataset}_${seed}/basic_${dataset}_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions
    ln -sf ../../../../../../${dataset}_16_564/basic_${dataset}_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions/test experiments/${dataset}_${seed}/basic_${dataset}_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions
    python -m llmcal ${dataset}_${seed} basic_${dataset}_0-shot_litgpt lm_tinyllama no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1
    for method in affine_scalar temp_scaling bias_only; do
        python -m llmcal ${dataset}_${seed} basic_${dataset}_0-shot_litgpt lm_tinyllama no_adaptation_bf16 $method --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    done
    python -m llmcal ${dataset}_${seed} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_10samples no_calibration --accelerator "gpu"
    python -m llmcal ${dataset}_${seed} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_10samples affine_scalar_train_on_val --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
done

dataset=sst2
for seed in 16_777 16_678 16_579 16_480 16_381; do
    mkdir -p experiments/${dataset}_${seed}/basic_${dataset}_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions
    ln -sf ../../../../../../${dataset}_16_564/basic_${dataset}_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions/test experiments/${dataset}_${seed}/basic_${dataset}_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions
    python -m llmcal ${dataset}_${seed} basic_${dataset}_0-shot_litgpt lm_tinyllama no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1
    for method in affine_scalar temp_scaling bias_only; do
        python -m llmcal ${dataset}_${seed} basic_${dataset}_0-shot_litgpt lm_tinyllama no_adaptation_bf16 $method --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    done
    python -m llmcal ${dataset}_${seed} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_20samples no_calibration --accelerator "gpu"
    python -m llmcal ${dataset}_${seed} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_20samples affine_scalar_train_on_val --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
done


dataset=agnews
for seed in 4_999 4_900 4_801 4_702 4_603; do
    mkdir -p experiments/${dataset}_${seed}/basic_${dataset}_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions
    ln -sf ../../../../../../${dataset}_16_738/basic_${dataset}_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions/test experiments/${dataset}_${seed}/basic_${dataset}_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions
    python -m llmcal ${dataset}_${seed} basic_${dataset}_0-shot_litgpt lm_tinyllama no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1
    for method in affine_scalar temp_scaling bias_only; do
        python -m llmcal ${dataset}_${seed} basic_${dataset}_0-shot_litgpt lm_tinyllama no_adaptation_bf16 $method --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    done
    python -m llmcal ${dataset}_${seed} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_10samples no_calibration --accelerator "gpu"
    python -m llmcal ${dataset}_${seed} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_10samples affine_scalar_train_on_val --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
done

dataset=agnews
for seed in 8_901 8_802 8_703 8_604 8_505; do
    mkdir -p experiments/${dataset}_${seed}/basic_${dataset}_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions
    ln -sf ../../../../../../${dataset}_16_738/basic_${dataset}_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions/test experiments/${dataset}_${seed}/basic_${dataset}_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions
    python -m llmcal ${dataset}_${seed} basic_${dataset}_0-shot_litgpt lm_tinyllama no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1
    for method in affine_scalar temp_scaling bias_only; do
        python -m llmcal ${dataset}_${seed} basic_${dataset}_0-shot_litgpt lm_tinyllama no_adaptation_bf16 $method --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    done
    python -m llmcal ${dataset}_${seed} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_20samples no_calibration --accelerator "gpu"
    python -m llmcal ${dataset}_${seed} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_20samples affine_scalar_train_on_val --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
done
