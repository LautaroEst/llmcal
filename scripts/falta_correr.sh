
# python -m llmcal sst2_256_812 basic_sst2_0-shot_litgpt lm_phi3 lora_500samples no_calibration --accelerator "gpu"

# agnews_4_962 no_adaptation_bf16
# agnews_4_962 lora_10samples
# me quedé en agnews256


for dataset in experiments.llama3_all/*; do
    for prompt in $dataset/*; do
        for model in $prompt/*; do
            founded_methods=()
            count=0
            for base_method in $model/*; do
                for cal_method in $base_method/*; do
                    if [ -e $cal_method/predictions/test ]; then
                        founded_methods+=($cal_method)
                        count=$((count+1))
                    fi
                done
            done
            echo $count $model
        done
    done
done