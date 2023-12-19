#/bin/bash


SCRIPTS_DIR="./scripts"
RESULTS_DIR="./results"

# MODEL="gpt2"
# MODEL="meta-llama/Llama-2-7b-hf"
# MODEL="gpt2-xl"

DATASET="tony_zhao/sst2"
echo ">>> Running ${MODEL} on ${DATASET}..."
python ${SCRIPTS_DIR}/run_dataset_on_model.py \
    --model_name ${MODEL} \
    --dataset_name ${DATASET} \
    --templates ${RESULTS_DIR}/templates/${DATASET}/00.json \
    --splits "train,test" \
    --num_samples "1000,None" \
    --output_dir ${RESULTS_DIR} \
    --save_embeddings \
    --random_state 9472

