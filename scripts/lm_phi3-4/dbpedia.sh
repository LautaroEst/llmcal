#!/bin/bash

### No adaptation + no calibration
python -m llmcal dbpedia_4_2951 basic_dbpedia_0-shot_litgpt lm_phi3 no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

# mkdir -p experiments/dbpedia_16_5641/basic_dbpedia_0-shot_litgpt/lm_phi3/no_adaptation_bf16/.cache/predictions
# ln -sf ../../../../../../dbpedia_4_2951/basic_dbpedia_0-shot_litgpt/lm_phi3/no_adaptation_bf16/.cache/predictions/test experiments/dbpedia_16_5641/basic_dbpedia_0-shot_litgpt/lm_phi3/no_adaptation_bf16/.cache/predictions
# python -m llmcal dbpedia_16_5641 basic_dbpedia_0-shot_litgpt lm_phi3 no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

# mkdir -p experiments/dbpedia_256_4931/basic_dbpedia_0-shot_litgpt/lm_phi3/no_adaptation_bf16/.cache/predictions
# ln -sf ../../../../../../dbpedia_4_2951/basic_dbpedia_0-shot_litgpt/lm_phi3/no_adaptation_bf16/.cache/predictions/test experiments/dbpedia_256_4931/basic_dbpedia_0-shot_litgpt/lm_phi3/no_adaptation_bf16/.cache/predictions
# python -m llmcal dbpedia_256_4931 basic_dbpedia_0-shot_litgpt lm_phi3 no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

mkdir -p experiments/dbpedia_2_4351/basic_dbpedia_0-shot_litgpt/lm_phi3/no_adaptation_bf16/.cache/predictions
ln -sf ../../../../../../dbpedia_4_2951/basic_dbpedia_0-shot_litgpt/lm_phi3/no_adaptation_bf16/.cache/predictions/test experiments/dbpedia_2_4351/basic_dbpedia_0-shot_litgpt/lm_phi3/no_adaptation_bf16/.cache/predictions
python -m llmcal dbpedia_2_4351 basic_dbpedia_0-shot_litgpt lm_phi3 no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

mkdir -p experiments/dbpedia_128_878/basic_dbpedia_0-shot_litgpt/lm_phi3/no_adaptation_bf16/.cache/predictions
ln -sf ../../../../../../dbpedia_4_2951/basic_dbpedia_0-shot_litgpt/lm_phi3/no_adaptation_bf16/.cache/predictions/test experiments/dbpedia_128_878/basic_dbpedia_0-shot_litgpt/lm_phi3/no_adaptation_bf16/.cache/predictions
python -m llmcal dbpedia_128_878 basic_dbpedia_0-shot_litgpt lm_phi3 no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

### No adaptation + affine scalar
python -m llmcal dbpedia_2_4351 basic_dbpedia_0-shot_litgpt lm_phi3 no_adaptation_bf16 affine_scalar \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

# python -m llmcal dbpedia_4_2951 basic_dbpedia_0-shot_litgpt lm_phi3 no_adaptation_bf16 affine_scalar \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

# python -m llmcal dbpedia_16_5641 basic_dbpedia_0-shot_litgpt lm_phi3 no_adaptation_bf16 affine_scalar \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

# python -m llmcal dbpedia_256_4931 basic_dbpedia_0-shot_litgpt lm_phi3 no_adaptation_bf16 affine_scalar \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

python -m llmcal dbpedia_128_878 basic_dbpedia_0-shot_litgpt lm_phi3 no_adaptation_bf16 affine_scalar \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

### No adaptation + temp scaling
python -m llmcal dbpedia_2_4351 basic_dbpedia_0-shot_litgpt lm_phi3 no_adaptation_bf16 temp_scaling \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

# python -m llmcal dbpedia_4_2951 basic_dbpedia_0-shot_litgpt lm_phi3 no_adaptation_bf16 temp_scaling \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

# python -m llmcal dbpedia_16_5641 basic_dbpedia_0-shot_litgpt lm_phi3 no_adaptation_bf16 temp_scaling \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

# python -m llmcal dbpedia_256_4931 basic_dbpedia_0-shot_litgpt lm_phi3 no_adaptation_bf16 temp_scaling \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

python -m llmcal dbpedia_128_878 basic_dbpedia_0-shot_litgpt lm_phi3 no_adaptation_bf16 temp_scaling \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

### No adaptation + bias only
python -m llmcal dbpedia_2_4351 basic_dbpedia_0-shot_litgpt lm_phi3 no_adaptation_bf16 bias_only \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

# python -m llmcal dbpedia_4_2951 basic_dbpedia_0-shot_litgpt lm_phi3 no_adaptation_bf16 bias_only \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

# python -m llmcal dbpedia_16_5641 basic_dbpedia_0-shot_litgpt lm_phi3 no_adaptation_bf16 bias_only \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

# python -m llmcal dbpedia_256_4931 basic_dbpedia_0-shot_litgpt lm_phi3 no_adaptation_bf16 bias_only \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

python -m llmcal dbpedia_128_878 basic_dbpedia_0-shot_litgpt lm_phi3 no_adaptation_bf16 bias_only \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

### Lora + no calibration
python -m llmcal dbpedia_2_4351 basic_dbpedia_0-shot_litgpt lm_phi3 lora_20samples no_calibration --accelerator "gpu"
# python -m llmcal dbpedia_4_2951 basic_dbpedia_0-shot_litgpt lm_phi3 lora_60samples no_calibration --accelerator "gpu"
# python -m llmcal dbpedia_16_5641 basic_dbpedia_0-shot_litgpt lm_phi3 lora_200samples no_calibration --accelerator "gpu"
# python -m llmcal dbpedia_256_4931 basic_dbpedia_0-shot_litgpt lm_phi3 lora_3500samples no_calibration --accelerator "gpu"
python -m llmcal dbpedia_128_878 basic_dbpedia_0-shot_litgpt lm_phi3 lora_1000samples no_calibration --accelerator "gpu"