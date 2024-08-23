# #!/bin/bash

# ### No adaptation + no calibration
python -m llmcal banking77_4_2951 basic_banking77_0-shot_litgpt lm_llama3 no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

# mkdir -p experiments/banking77_16_5641/basic_banking77_0-shot_litgpt/lm_llama3/no_adaptation_bf16/.cache/predictions
# ln -sf ../../../../../../banking77_4_2951/basic_banking77_0-shot_litgpt/lm_llama3/no_adaptation_bf16/.cache/predictions/test experiments/banking77_16_5641/basic_banking77_0-shot_litgpt/lm_llama3/no_adaptation_bf16/.cache/predictions
# python -m llmcal banking77_16_5641 basic_banking77_0-shot_litgpt lm_llama3 no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

mkdir -p experiments/banking77_64_912/basic_banking77_0-shot_litgpt/lm_llama3/no_adaptation_bf16/.cache/predictions
ln -sf ../../../../../../banking77_4_2951/basic_banking77_0-shot_litgpt/lm_llama3/no_adaptation_bf16/.cache/predictions/test experiments/banking77_64_912/basic_banking77_0-shot_litgpt/lm_llama3/no_adaptation_bf16/.cache/predictions
python -m llmcal banking77_64_912 basic_banking77_0-shot_litgpt lm_llama3 no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

mkdir -p experiments/banking77_1_858/basic_banking77_0-shot_litgpt/lm_llama3/no_adaptation_bf16/.cache/predictions
ln -sf ../../../../../../banking77_4_2951/basic_banking77_0-shot_litgpt/lm_llama3/no_adaptation_bf16/.cache/predictions/test experiments/banking77_1_858/basic_banking77_0-shot_litgpt/lm_llama3/no_adaptation_bf16/.cache/predictions
python -m llmcal banking77_1_858 basic_banking77_0-shot_litgpt lm_llama3 no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

### No adaptation + affine scalar
python -m llmcal banking77_1_858 basic_banking77_0-shot_litgpt lm_llama3 no_adaptation_bf16 affine_scalar \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

# python -m llmcal banking77_4_2951 basic_banking77_0-shot_litgpt lm_llama3 no_adaptation_bf16 affine_scalar \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

# python -m llmcal banking77_16_5641 basic_banking77_0-shot_litgpt lm_llama3 no_adaptation_bf16 affine_scalar \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

python -m llmcal banking77_64_912 basic_banking77_0-shot_litgpt lm_llama3 no_adaptation_bf16 affine_scalar \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

### No adaptation + temp scaling
python -m llmcal banking77_1_858 basic_banking77_0-shot_litgpt lm_llama3 no_adaptation_bf16 temp_scaling \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

# python -m llmcal banking77_4_2951 basic_banking77_0-shot_litgpt lm_llama3 no_adaptation_bf16 temp_scaling \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

# python -m llmcal banking77_16_5641 basic_banking77_0-shot_litgpt lm_llama3 no_adaptation_bf16 temp_scaling \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

python -m llmcal banking77_64_912 basic_banking77_0-shot_litgpt lm_llama3 no_adaptation_bf16 temp_scaling \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

### No adaptation + bias only
python -m llmcal banking77_1_858 basic_banking77_0-shot_litgpt lm_llama3 no_adaptation_bf16 bias_only \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30


# python -m llmcal banking77_4_2951 basic_banking77_0-shot_litgpt lm_llama3 no_adaptation_bf16 bias_only \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

# python -m llmcal banking77_16_5641 basic_banking77_0-shot_litgpt lm_llama3 no_adaptation_bf16 bias_only \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

python -m llmcal banking77_64_912 basic_banking77_0-shot_litgpt lm_llama3 no_adaptation_bf16 bias_only \
    --accelerator "gpu" \
    --batch_size 1 \
    --calibration.max_ls 40 \
    --calibration.learning_rate 1e-2 \
    --calibration.accelerator "cpu" \
    --calibration.max_epochs 30

# ### Lora + no calibration
python -m llmcal banking77_1_858 basic_banking77_0-shot_litgpt lm_llama3 lora_60samples no_calibration --accelerator "gpu"
# python -m llmcal banking77_4_2951 basic_banking77_0-shot_litgpt lm_llama3 lora_200samples no_calibration --accelerator "gpu"
# python -m llmcal banking77_16_5641 basic_banking77_0-shot_litgpt lm_llama3 lora_1000samples no_calibration --accelerator "gpu"
python -m llmcal banking77_64_912 basic_banking77_0-shot_litgpt lm_llama3 lora_5000samples no_calibration --accelerator "gpu"