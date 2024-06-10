# #!/bin/bash

# ### No adaptation + no calibration
# python -m llmcal banking77_4_926 basic_banking77_0-shot_litgpt lm_tinyllama no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

# mkdir -p experiments/banking77_16_564/basic_banking77_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions
# ln -sf ../../../../../../banking77_4_926/basic_banking77_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions/test experiments/banking77_16_564/basic_banking77_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions
# python -m llmcal banking77_16_564 basic_banking77_0-shot_litgpt lm_tinyllama no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

# mkdir -p experiments/banking77_64_131/basic_banking77_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions
# ln -sf ../../../../../../banking77_4_926/basic_banking77_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions/test experiments/banking77_64_131/basic_banking77_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/.cache/predictions
# python -m llmcal banking77_64_131 basic_banking77_0-shot_litgpt lm_tinyllama no_adaptation_bf16 no_calibration --accelerator "gpu" --batch_size 1

# ### No adaptation + affine scalar
# python -m llmcal banking77_4_926 basic_banking77_0-shot_litgpt lm_tinyllama no_adaptation_bf16 affine_scalar \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

# python -m llmcal banking77_16_564 basic_banking77_0-shot_litgpt lm_tinyllama no_adaptation_bf16 affine_scalar \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

# python -m llmcal banking77_64_131 basic_banking77_0-shot_litgpt lm_tinyllama no_adaptation_bf16 affine_scalar \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

# ### No adaptation + temp scaling
# python -m llmcal banking77_4_926 basic_banking77_0-shot_litgpt lm_tinyllama no_adaptation_bf16 temp_scaling \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

# python -m llmcal banking77_16_564 basic_banking77_0-shot_litgpt lm_tinyllama no_adaptation_bf16 temp_scaling \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

# python -m llmcal banking77_64_131 basic_banking77_0-shot_litgpt lm_tinyllama no_adaptation_bf16 temp_scaling \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

# ### No adaptation + bias only
# python -m llmcal banking77_4_926 basic_banking77_0-shot_litgpt lm_tinyllama no_adaptation_bf16 bias_only \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

# python -m llmcal banking77_16_564 basic_banking77_0-shot_litgpt lm_tinyllama no_adaptation_bf16 bias_only \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

# python -m llmcal banking77_64_131 basic_banking77_0-shot_litgpt lm_tinyllama no_adaptation_bf16 bias_only \
#     --accelerator "gpu" \
#     --batch_size 1 \
#     --calibration.max_ls 40 \
#     --calibration.learning_rate 1e-2 \
#     --calibration.accelerator "cpu" \
#     --calibration.max_epochs 30

# ### Lora + no calibration
python -m llmcal banking77_4_926 basic_banking77_0-shot_litgpt lm_tinyllama lora_200samples no_calibration --accelerator "gpu"
# python -m llmcal banking77_16_564 basic_banking77_0-shot_litgpt lm_tinyllama lora_1000samples no_calibration --accelerator "gpu"
# python -m llmcal banking77_64_131 basic_banking77_0-shot_litgpt lm_tinyllama lora_5000samples no_calibration --accelerator "gpu"