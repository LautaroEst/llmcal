#/bin/bash


SCRIPTS_DIR="./scripts"
RESULTS_DIR="./results"

MODEL="gpt2"
# MODEL="meta-llama/Llama-2-7b-hf"
DATASETS=("tony_zhao/sst2" )

for DATASET in ${DATASETS[@]}; do
    echo ">>> Running ${MODEL} on ${DATASET}..."
    python ${SCRIPTS_DIR}/run_dataset_on_model.py \
        --model_name ${MODEL} \
        --dataset_name ${DATASET} \
        --templates_path ${RESULTS_DIR}/templates \
        --splits "train,test" \
        --num_samples "600,None" \
        --output_dir ${RESULTS_DIR} \
        --save_embeddings \
        --batch_size 4 \
        --random_state 9472
done