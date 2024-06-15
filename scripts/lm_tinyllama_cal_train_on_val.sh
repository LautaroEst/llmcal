# #!/bin/bash

# dataset=sst2
# for suffix in 8_639 8_923 8_932 8_6391 8_9322; do
#     for method in affine_scalar_train_on_val ; do
#         python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_10samples $method --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
#     done
# done

# dataset=sst2
# for suffix in 16_564 16_738 16_783 16_5641 16_7832; do
#     for method in affine_scalar_train_on_val ; do
#         python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_20samples $method --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
#     done
# done

# dataset=sst2
# for suffix in 32_1564 32_1738 32_1783 32_15641 32_17832; do
#     for method in affine_scalar_train_on_val ; do
#         python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_40samples $method --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
#     done
# done

# dataset=sst2
# for suffix in 256_493 256_812 256_821 256_4931 256_8212; do
#     for method in affine_scalar_train_on_val ; do
#         python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_500samples $method --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
#     done
# done

# dataset=agnews
# for suffix in 4_295 4_926 4_962 4_2951 4_9622; do
#     for method in affine_scalar_train_on_val ; do
#         python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_10samples $method --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
#     done
# done

# dataset=agnews
# for suffix in 8_639 8_923 8_932 8_6391 8_9322; do
#     for method in affine_scalar_train_on_val ; do
#         python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_20samples $method --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
#     done
# done

# dataset=agnews
# for suffix in 16_564 16_738 16_783 16_5641 16_7832; do
#     for method in affine_scalar_train_on_val ; do
#         python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_60samples $method --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
#     done
# done

# dataset=agnews
# for suffix in 256_493 256_812 256_821 256_4931 256_8212; do
#     for method in affine_scalar_train_on_val ; do
#         python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_1000samples $method --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
#     done
# done


# dataset=dbpedia
# for suffix in 2_435 2_927 2_972 2_4351 2_9722; do
#     for method in affine_scalar_train_on_val ; do
#         python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_20samples $method --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
#     done
# done

# dataset=dbpedia
# for suffix in 4_295 4_926 4_962 4_2951 4_9622; do
#     for method in affine_scalar_train_on_val ; do
#         python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_60samples $method --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
#     done
# done

# dataset=dbpedia
# for suffix in 16_564 16_738 16_783 16_5641 16_7832; do
#     for method in affine_scalar_train_on_val ; do
#         python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_200samples $method --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
#     done
# done

# dataset=dbpedia
# for suffix in 256_493 256_812 256_821 256_4931 256_8212; do
#     for method in affine_scalar_train_on_val ; do
#         python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_3500samples $method --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
#     done
# done

# dataset=20newsgroups
# for suffix in 2_435 2_927 2_972 2_4351 2_9722; do
#     for method in affine_scalar_train_on_val ; do
#         python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_40samples $method --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
#     done
# done

# dataset=20newsgroups
# for suffix in 4_295 4_926 4_962 4_2951 4_9622; do
#     for method in affine_scalar_train_on_val ; do
#         python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_60samples $method --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
#     done
# done

# dataset=20newsgroups
# for suffix in 16_564 16_738 16_783 16_5641 16_7832; do
#     for method in affine_scalar_train_on_val ; do
#         python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_200samples $method --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
#     done
# done

# dataset=20newsgroups
# for suffix in 256_493 256_812 256_821 256_4931 256_8212; do
#     for method in affine_scalar_train_on_val ; do
#         python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_3500samples $method --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
#     done
# done

dataset=banking77
# for suffix in 4_295 4_926 4_962 4_2951 4_9622; do
for suffix in 4_2951 ; do
    for method in affine_scalar_train_on_val ; do
        python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_200samples no_calibration --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
        python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_200samples $method --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    done
done

dataset=banking77
# for suffix in 16_564 16_738 16_783 16_5641 16_7832; do
for suffix in 16_5641 ; do
    for method in affine_scalar_train_on_val ; do
        python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_1000samples no_calibration --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
        python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_1000samples $method --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    done
done

dataset=banking77
# for suffix in 64_131 64_893 64_933 64_888 64_912; do
for suffix in 64_912 ; do
    for method in affine_scalar_train_on_val ; do
        python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_5000samples no_calibration --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
        python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama lora_5000samples $method --accelerator "gpu" --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    done
done
