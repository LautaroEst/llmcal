#/bin/bash

SCRIPTS_DIR="./scripts"
RESULTS_DIR="./results"

# Model:
# MODEL="gpt2"
# MODEL="meta-llama/Llama-2-7b-hf"
MODEL="gpt2-xl"

# Template:
TEMPLATE_NAME="00.json"


# DATASETS=("tony_zhao/trec" "tony_zhao/sst2" "tony_zhao/agnews" "tony_zhao/dbpedia")
# DATASETS=("tony_zhao/sst2" )
DATASETS=("refind" )
for DATASET in ${DATASETS[@]}; do
    python ${SCRIPTS_DIR}/run_dataset_on_model.py \
        --model_name ${MODEL} \
        --dataset_name ${DATASET} \
        --templates ${RESULTS_DIR}/templates/${DATASET}/${TEMPLATE_NAME} \
        --splits "train,dev,test" \
        --num_samples "4000,None,None" \
        --output_dir ${RESULTS_DIR} \
        --save_embeddings \
        --random_state 9472 \
        --accelerator "cpu" \
        --devices "1"
done


# # DATASETS=("glue/cola" "glue/sst2" "glue/mrpc" "glue/qqp" "glue/mnli" "glue/qnli" "glue/rte" "glue/wnli")
# DATASETS=("glue/sst2" )
# for DATASET in ${DATASETS[@]}; do
#     python ${SCRIPTS_DIR}/run_dataset_on_model.py \
#         --model_name ${MODEL} \
#         --dataset_name ${DATASET} \
#         --templates ${RESULTS_DIR}/templates/${DATASET}/${TEMPLATE_NAME} \
#         --splits "train,validation" \
#         --num_samples "4000,None" \
#         --output_dir ${RESULTS_DIR} \
#         --save_embeddings \
#         --random_state 3945
# done

echo =============
echo End of script
echo =============
