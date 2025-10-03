
# This script sets up the environment for running experiments with different models and datasets.
# If you want to re-run the models, please set the environment variable CHECKPOINTS_DIR to the directory where you want to store the checkpoints.
# Then, ensure you have the Hugging Face token stored in a file named hf_token.txt in the same directory as this script (it is already hidden in the .gitignore file).
# Then, select the model you want to run by setting the variable `model` to one of the supported models in the `model2checkpoint` dictionary.

# CHECKPOINTS_DIR=./checkpoints
HF_TOKEN=$(cat hf_token.txt)
model="llama3.2-1b-instruct"
# model="qwen2.5-7b-instruct"

# Reproducibility
base_seed=2834
declare -A dataset2nseeds=(
    ["sst2"]=9
    ["agnews"]=9
    ["dbpedia"]=5
    ["20newsgroups"]=5
    ["banking77"]=5
)
num_seeds=9

# Supported models
declare -A model2checkpoint=(
    ["llama3.2-1b"]="meta-llama/Llama-3.2-1B"
    ["llama3.2-1b-instruct"]="meta-llama/Llama-3.2-1B-Instruct"
    ["qwen2.5-7b"]="Qwen/Qwen2.5-7B"
    ["qwen2.5-7b-instruct"]="Qwen/Qwen2.5-7B-Instruct"
)

#### UNCOMMENT THE FOLLOWING LINES TO DOWNLOAD THE CHECKPOINTS #####

# mkdir -p $CHECKPOINTS_DIR
# if [ ! -d $CHECKPOINTS_DIR/${model2checkpoint[$model]} ]; then
#     litgpt download ${model2checkpoint[$model]} --checkpoint_dir $CHECKPOINTS_DIR --access_token $HF_TOKEN
#     rm -rf $CHECKPOINTS_DIR/${model2checkpoint[$model]}/*.bin
# fi
# if [ ! -z ${model2checkpoint[${model}-instruct]} ]; then
#     if [ ! -d $CHECKPOINTS_DIR/${model2checkpoint[${model}-instruct]} ]; then
#         litgpt download ${model2checkpoint[${model}-instruct]} --checkpoint_dir $CHECKPOINTS_DIR --access_token $HF_TOKEN
#         rm -rf $CHECKPOINTS_DIR/${model2checkpoint[${model}-instruct]}/*.bin
#     fi
# fi

#####################################################################


# Datasets
declare -a DATASETS=(20newsgroups dbpedia sst2 agnews banking77)

# Train sizes
declare -a FACTORS=(16 32 64 128 256)

# Test sizes
declare -A dataset2testsize=(
    ["sst2"]=400
    ["agnews"]=400
    ["dbpedia"]=700
    ["20newsgroups"]=800
    ["banking77"]=1000
)

max_seq_length=2048
inference_max_seq_len=20000

export CUDA_VISIBLE_DEVICES=1