#!/bin/bash -ex

source ./scripts/env.sh
declare -a models=(llama3.2-1b-instruct qwen2.5-7b-instruct)
declare -a DATASETS=(sst2 agnews dbpedia 20newsgroups banking77)
metrics=(nce cal_err ece)
overwrite=true

for model in "${models[@]}"; do

    # for metric in "${metrics[@]}"; do

    #     # Compute results:
    #     results_path=outputs/results_paper/$model/$metric.jsonl
    #     if [ -f $results_path ] && [ $overwrite = false ]; then
    #         echo "Results already computed. Skipping."
    #     else
    #         mkdir -p $(dirname $results_path)
    #         python -m llmcal.scripts.compute_matched_results \
    #             --metric $metric \
    #             --finetuning_root_results_dirs outputs/finetune_lora/$model/ \
    #             --output_path $results_path \
    #             --reduced \
    #             --no_adaptation_root_results_dirs outputs/no_adaptation/$model \
    #             --lora_plus_cal_root_results_dirs "outputs/lora_plus_dpcal/$model,outputs/lora_plus_tempscaling/$model,outputs/lora_plus_biasshift/$model,outputs/lora_plus_vectorscaling/$model" \
    #             --lora_plus_cal_naive_root_results_dirs "outputs/lora_plus_dpcal_naive/$model,outputs/lora_plus_tempscaling_naive/$model" \
    #             --trainontest_root_results_dirs outputs/lora_plus_dpcal_trainontest/$model,outputs/lora_plus_tempscaling_trainontest/$model \
    #             --cal_root_results_dirs outputs/calibration/$model
    #     fi
    # done

    # Training samples:
    samples_plots_path="outputs/results_paper/$model/metric_vs_samples/adaptation_performance_$model.png"
    mkdir -p $(dirname $samples_plots_path)
    python -m llmcal.scripts.results_vs_samples \
        --datasets "${DATASETS[*]}" \
        --metrics "${metrics[*]}" \
        --sizes "${FACTORS[*]}" \
        --methods_config "./configs/methods_final.yaml" \
        --results_dir outputs/results_paper/$model \
        --output_path $samples_plots_path \
        --intervals \
        --methods "no_adaptation lora_1.0 lora_1.0_no_es dp_calibration lora_1.0_no_es_plus_tempscaling"


    samples_table_path="outputs/results_paper/$model/results_table/$model.tex"
    mkdir -p $(dirname $samples_table_path)
    python -m llmcal.scripts.results_table \
        --datasets "${DATASETS[*]}" \
        --metrics "${metrics[*]}" \
        --sizes "${FACTORS[*]}" \
        --methods_config "./configs/methods_final.yaml" \
        --results_dir outputs/results_paper/$model \
        --output_path $samples_table_path \
        --methods "no_adaptation dirichlet_fixed_diag temp_scaling vector_scaling bias_shift dp_calibration lora_0.7 lora_1.0 lora_1.0_no_es lora_1.0_no_es_plus_tempscaling lora_1.0_no_es_plus_dpcal lora_1.0_no_es_plus_biasshift lora_1.0_no_es_plus_vectorscaling"

    samples_bars_path="outputs/results_paper/$model/results_bars/$model.pdf"
    mkdir -p $(dirname $samples_bars_path)
    python -m llmcal.scripts.results_bars \
        --datasets "${DATASETS[*]}" \
        --metrics "${metrics[*]}" \
        --sizes "${FACTORS[*]}" \
        --methods_config "./configs/methods_final.yaml" \
        --results_dir outputs/results_paper/$model \
        --output_path $samples_bars_path \
        --methods "dirichlet_fixed_diag dp_calibration bias_shift temp_scaling lora_0.7 lora_1.0 lora_1.0_no_es lora_1.0_no_es_plus_dpcal lora_1.0_no_es_plus_biasshift lora_1.0_no_es_plus_tempscaling no_adaptation"
done

samples_plots_path="outputs/results_paper/all_models/comparison.png"
results_dirs=$(for model in "${models[@]}"; do echo "outputs/results_paper/$model"; done | tr '\n' ' ')
results_dirs=${results_dirs::-1}

mkdir -p $(dirname $samples_plots_path)
python -m llmcal.scripts.compare_models \
    --datasets "${DATASETS[*]}" \
    --metrics "${metrics[*]}" \
    --sizes "${FACTORS[*]}" \
    --methods_config "./configs/methods_final.yaml" \
    --output_path $samples_plots_path \
    --models "${models[*]}" \
    --results_dirs "$results_dirs" \
    --intervals \
    --methods "no_adaptation dp_calibration lora_1.0_no_es lora_1.0_no_es_plus_tempscaling"

