#/bin/bash


SCRIPTS_DIR="./scripts"
RESULTS_DIR="./results"

MODEL="gpt2-xl"
# MODEL="meta-llama/Llama-2-7b-hf"

DATASETS=("glue/cola" "glue/sst2" "glue/mrpc" "glue/qqp" "glue/mnli" "glue/qnli" "glue/rte" "glue/wnli")

for DATASET in ${DATASETS[@]}; do
    python ${SCRIPTS_DIR}/run_dataset_on_model.py \
        --model_name ${MODEL} \
        --dataset_name ${DATASET} \
        --templates ${RESULTS_DIR}/templates/${DATASET}/00.json \
        --splits "train,validation" \
        --num_samples "4000,None" \
        --output_dir ${RESULTS_DIR} \
        --save_embeddings \
        --random_state 3945
done

echo =============
echo End of script
echo =============