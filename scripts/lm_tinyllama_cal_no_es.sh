# #!/bin/bash

dataset=sst2
for suffix in 8_639 8_923 8_932 8_6391 8_9322; do
    for method in affine_scalar_no_es ; do
        python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama no_adaptation_bf16 $method --accelerator "gpu" --val_prop 0 --total_train_samples 16 --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    done
done

dataset=agnews
for suffix in 8_639 8_923 8_932 8_6391 8_9322; do
    for method in affine_scalar_no_es ; do
        python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama no_adaptation_bf16 $method --accelerator "gpu" --val_prop 0 --total_train_samples 32 --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    done
done

dataset=dbpedia
for suffix in 2_435 2_927 2_972 2_4351 2_9722; do
    for method in affine_scalar_no_es ; do
        python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama no_adaptation_bf16 $method --accelerator "gpu" --val_prop 0 --total_train_samples 28 --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    done
done

dataset=20newsgroups
for suffix in 2_435 2_927 2_972 2_9722; do
    for method in affine_scalar_no_es ; do
        python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama no_adaptation_bf16 $method --accelerator "gpu" --val_prop 0 --total_train_samples 40 --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    done
done

dataset=sst2
for suffix in 16_564 16_738 16_783 16_5641 16_7832; do
    for method in affine_scalar_no_es ; do
        python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama no_adaptation_bf16 $method --accelerator "gpu" --val_prop 0 --total_train_samples 32 --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    done
done

dataset=agnews
for suffix in 16_564 16_738 16_783 16_5641 16_7832; do
    for method in affine_scalar_no_es ; do
        python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama no_adaptation_bf16 $method --accelerator "gpu" --val_prop 0 --total_train_samples 64 --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    done
done

dataset=dbpedia
for suffix in 4_295 4_926 4_962 4_2951 4_9622; do
    for method in affine_scalar_no_es ; do
        python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama no_adaptation_bf16 $method --accelerator "gpu" --val_prop 0 --total_train_samples 224 --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    done
done

dataset=20newsgroups
for suffix in 4_295 4_926 4_962 4_9622; do
    for method in affine_scalar_no_es ; do
        python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama no_adaptation_bf16 $method --accelerator "gpu" --val_prop 0 --total_train_samples 80 --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    done
done

dataset=sst2
for suffix in 256_493 256_812 256_821 256_4931 256_8212; do
    for method in affine_scalar_no_es ; do
        python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama no_adaptation_bf16 $method --accelerator "gpu" --val_prop 0 --total_train_samples 512 --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    done
done

dataset=agnews
for suffix in 256_493 256_812 256_821 256_4931 256_8212; do
    for method in affine_scalar_no_es ; do
        python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama no_adaptation_bf16 $method --accelerator "gpu" --val_prop 0 --total_train_samples 4096 --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    done
done

dataset=dbpedia
for suffix in 256_493 256_812 256_821 256_4931 256_8212; do
    for method in affine_scalar_no_es ; do
        python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama no_adaptation_bf16 $method --accelerator "gpu" --val_prop 0 --total_train_samples 3584 --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    done
done

dataset=20newsgroups
for suffix in 256_493 256_812 256_8212; do
    for method in affine_scalar_no_es ; do
        python -m llmcal ${dataset}_${suffix} basic_${dataset}_0-shot_litgpt lm_tinyllama no_adaptation_bf16 $method --accelerator "gpu" --val_prop 0 --total_train_samples 5120 --batch_size 1 --calibration.max_ls 40 --calibration.learning_rate 1e-2 --calibration.accelerator "cpu" --calibration.max_epochs 30
    done
done
