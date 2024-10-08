#!/bin/bash

### No adaptation + no calibration
# python -m llmcal dbpedia_4_295 basic_dbpedia_0-shot_litgpt lm_tinyllama no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

mkdir -p experiments/dbpedia_8_338/basic_dbpedia_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions
ln -sf ../../../../../../dbpedia_4_295/basic_dbpedia_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions/test experiments/dbpedia_8_338/basic_dbpedia_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions
python -m llmcal dbpedia_8_338 basic_dbpedia_0-shot_litgpt lm_tinyllama no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

# mkdir -p experiments/dbpedia_16_738/basic_dbpedia_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions
# ln -sf ../../../../../../dbpedia_4_295/basic_dbpedia_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions/test experiments/dbpedia_16_738/basic_dbpedia_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions
# python -m llmcal dbpedia_16_738 basic_dbpedia_0-shot_litgpt lm_tinyllama no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

# mkdir -p experiments/dbpedia_256_493/basic_dbpedia_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions
# ln -sf ../../../../../../dbpedia_4_295/basic_dbpedia_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions/test experiments/dbpedia_256_493/basic_dbpedia_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions
# python -m llmcal dbpedia_256_493 basic_dbpedia_0-shot_litgpt lm_tinyllama no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

mkdir -p experiments/dbpedia_128_129/basic_dbpedia_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions
ln -sf ../../../../../../dbpedia_4_295/basic_dbpedia_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions/test experiments/dbpedia_128_129/basic_dbpedia_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions
python -m llmcal dbpedia_128_129 basic_dbpedia_0-shot_litgpt lm_tinyllama no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

# mkdir -p experiments/dbpedia_2_927/basic_dbpedia_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions
# ln -sf ../../../../../../dbpedia_4_295/basic_dbpedia_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions/test experiments/dbpedia_2_927/basic_dbpedia_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions
# python -m llmcal dbpedia_2_927 basic_dbpedia_0-shot_litgpt lm_tinyllama no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

### No adaptation + affine scalar
# python -m llmcal dbpedia_2_927 basic_dbpedia_0-shot_litgpt lm_tinyllama no_adaptation_bf16 affine_scalar \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

# python -m llmcal dbpedia_4_295 basic_dbpedia_0-shot_litgpt lm_tinyllama no_adaptation_bf16 affine_scalar \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

python -m llmcal dbpedia_8_338 basic_dbpedia_0-shot_litgpt lm_tinyllama no_adaptation_bf16 affine_scalar \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

# python -m llmcal dbpedia_16_738 basic_dbpedia_0-shot_litgpt lm_tinyllama no_adaptation_bf16 affine_scalar \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

# python -m llmcal dbpedia_256_493 basic_dbpedia_0-shot_litgpt lm_tinyllama no_adaptation_bf16 affine_scalar \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

python -m llmcal dbpedia_128_129 basic_dbpedia_0-shot_litgpt lm_tinyllama no_adaptation_bf16 affine_scalar \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

### No adaptation + temp scaling
# python -m llmcal dbpedia_2_927 basic_dbpedia_0-shot_litgpt lm_tinyllama no_adaptation_bf16 temp_scaling \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

# python -m llmcal dbpedia_4_295 basic_dbpedia_0-shot_litgpt lm_tinyllama no_adaptation_bf16 temp_scaling \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

python -m llmcal dbpedia_8_338 basic_dbpedia_0-shot_litgpt lm_tinyllama no_adaptation_bf16 temp_scaling \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

# python -m llmcal dbpedia_16_738 basic_dbpedia_0-shot_litgpt lm_tinyllama no_adaptation_bf16 temp_scaling \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

# python -m llmcal dbpedia_256_493 basic_dbpedia_0-shot_litgpt lm_tinyllama no_adaptation_bf16 temp_scaling \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
    # --calibration.max_epochs 30

python -m llmcal dbpedia_128_129 basic_dbpedia_0-shot_litgpt lm_tinyllama no_adaptation_bf16 temp_scaling \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

### No adaptation + bias only
# python -m llmcal dbpedia_2_927 basic_dbpedia_0-shot_litgpt lm_tinyllama no_adaptation_bf16 bias_only \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

# python -m llmcal dbpedia_4_295 basic_dbpedia_0-shot_litgpt lm_tinyllama no_adaptation_bf16 bias_only \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

python -m llmcal dbpedia_8_338 basic_dbpedia_0-shot_litgpt lm_tinyllama no_adaptation_bf16 bias_only \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

# python -m llmcal dbpedia_16_738 basic_dbpedia_0-shot_litgpt lm_tinyllama no_adaptation_bf16 bias_only \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

# python -m llmcal dbpedia_256_493 basic_dbpedia_0-shot_litgpt lm_tinyllama no_adaptation_bf16 bias_only \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

python -m llmcal dbpedia_128_129 basic_dbpedia_0-shot_litgpt lm_tinyllama no_adaptation_bf16 bias_only \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

### Lora + no calibration
# python -m llmcal dbpedia_2_927 basic_dbpedia_0-shot_litgpt lm_tinyllama lora_20samples no_calibration --accelerator "gpu"
# python -m llmcal dbpedia_4_295 basic_dbpedia_0-shot_litgpt lm_tinyllama lora_60samples no_calibration --accelerator "gpu"
python -m llmcal dbpedia_8_338 basic_dbpedia_0-shot_litgpt lm_tinyllama lora_200samples no_calibration --accelerator "gpu"
# python -m llmcal dbpedia_16_738 basic_dbpedia_0-shot_litgpt lm_tinyllama lora_200samples no_calibration --accelerator "gpu"
# python -m llmcal dbpedia_256_493 basic_dbpedia_0-shot_litgpt lm_tinyllama lora_3500samples no_calibration --accelerator "gpu"
python -m llmcal dbpedia_128_129 basic_dbpedia_0-shot_litgpt lm_tinyllama lora_1000samples no_calibration --accelerator "gpu"