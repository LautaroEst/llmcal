#!/bin/bash -ex

source ./scripts/env.sh

metric=ner

overwrite=true
results_path=outputs/results/$model/$metric.jsonl
if [ -f $results_path ] && [ $overwrite = false ]; then
    echo "Results already computed. Skipping."
else
    mkdir -p $(dirname $results_path)
    python -m llmcal.scripts.compute_results \
        --metric $metric \
        --root_results_dir outputs/finetune_lora/$model/ \
        --output_path $results_path
fi

# Training samples:
samples_plots_dir=outputs/results/$model/metric_vs_samples
mkdir -p $samples_plots_dir
python -m llmcal.scripts.plot_metric_vs_samples \
    --datasets "${DATASETS[*]}" \
    --metric $metric \
    --sizes "${FACTORS[*]}" \
    --methods "lora_ans lora_ans_no_es" \
    --results_path $results_path \
    --output_dir $samples_plots_dir