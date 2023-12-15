#/bin/bash


SCRIPTS_DIR="./scripts"
RESULTS_DIR="./results"

MODEL="gpt2"
# MODEL="meta-llama/Llama-2-7b-hf"

SEEDS=(17823 234 1023 923 28394)
DATASETS=("glue/sst2" )

for SEED in ${SEEDS[@]}; do
    for DATASET in ${DATASETS[@]}; do
        echo ">>> Running ${MODEL} on ${DATASET} with seed ${SEED}..."
        python ${SCRIPTS_DIR}/run_dataset_on_model.py \
            --model_name ${MODEL} \
            --dataset_name ${DATASET} \
            --templates_path ${RESULTS_DIR}/templates \
            --splits "train,validation" \
            --num_samples "1000,None" \
            --output_dir ${RESULTS_DIR} \
            --save_embeddings \
            --batch_size 32 \
            --seed ${SEED}
    done
done