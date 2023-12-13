#/bin/bash


SCRIPTS_DIR="./scripts"
RESULTS_DIR="./results"

model_name="gpt2"
# model_name = "meta-llama/Llama-2-7b-hf"

SEED=84283

python ${SCRIPTS_DIR}/run_dataset_on_model.py \
    --model_name ${model_name} \
    --dataset_name "glue/sst2" \
    --output_dir ${RESULTS_DIR} \
    --batch_size 32 \
    --seed ${SEED}