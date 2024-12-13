#!/bin/bash -ex

source ./scripts/env.sh

metrics=(nce ner nbrier)
overwrite=true

for metric in "${metrics[@]}"; do

    # Compute results:
    results_path=outputs/results/$model/$metric.jsonl
    if [ -f $results_path ] && [ $overwrite = false ]; then
        echo "Results already computed. Skipping."
    else
        mkdir -p $(dirname $results_path)
        python -m llmcal.scripts.compute_matched_results \
            --metric $metric \
            --finetuning_root_results_dirs outputs/finetune_lora/$model/ \
            --lora_plus_cal_root_results_dirs "outputs/lora_plus_dpcal/$model,outputs/lora_plus_tempscaling/$model" \
            --cal_root_results_dirs outputs/calibration/$model \
            --trainontest_root_results_dirs outputs/lora_plus_dpcal_trainontest/$model,outputs/lora_plus_tempscaling_trainontest/$model \
            --output_path $results_path \
            --reduced 
    fi

    # Training samples:
    samples_plots_dir=outputs/results/$model/metric_vs_samples
    mkdir -p $samples_plots_dir
    python -m llmcal.scripts.results_vs_samples \
        --datasets "${DATASETS[*]}" \
        --metric $metric \
        --sizes "${FACTORS[*]}" \
        --methods "lora_0.7 lora_0.7_no_es lora_0.7_plus_tempscaling lora_0.7_no_es_plus_tempscaling lora_1.0_plus_dpcal_trainontest" \
        --methods_config "./configs/methods.yaml" \
        --results_path $results_path \
        --output_dir $samples_plots_dir
done
# --methods "lora_1.0 lora_1.0_no_es lora_1.0_plus_dpcal lora_1.0_plus_tempscaling lora_1.0_plus_dpcal_trainontest lora_1.0_plus_tempscaling_trainontest" \