#/bin/bash


SCRIPTS_DIR="./scripts"
RESULTS_DIR="./results"

# MODEL="gpt2"
# MODEL="meta-llama/Llama-2-7b-hf"
MODEL="gpt2-xl"

# DATASETS=("tony_zhao/trec" "tony_zhao/sst2" "tony_zhao/agnews" "tony_zhao/dbpedia")
DATASETS=("tony_zhao/sst2" )

for DATASET in ${DATASETS[@]}; do
    python ${SCRIPTS_DIR}/run_dataset_on_model.py \
        --model_name ${MODEL} \
        --dataset_name ${DATASET} \
        --templates ${RESULTS_DIR}/templates/${DATASET}/00.json \
        --splits "train,test" \
        --num_samples "4000,None" \
        --output_dir ${RESULTS_DIR} \
        --save_embeddings \
        --random_state 9472
done

echo =============
echo End of script
echo =============
