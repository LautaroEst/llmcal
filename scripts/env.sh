CHECKPOINTS_DIR=../llmcal2/outputs/checkpoints
HF_TOKEN=$(cat hf_token.txt)
model="llama3.2-1b-instruct"
# model="pythia-14m"
# model=tinyllama

# Reproducibility
base_seed=2834
num_seeds=5

# Supported models
declare -A model2checkpoint=(
    ["pythia-14m"]="EleutherAI/pythia-14m"
    ["llama3.2-1b"]="meta-llama/Llama-3.2-1B"
    ["llama3.2-1b-instruct"]="meta-llama/Llama-3.2-1B-Instruct"
    ["tinyllama"]="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
)
mkdir -p $CHECKPOINTS_DIR
if [ ! -d $CHECKPOINTS_DIR/${model2checkpoint[$model]} ]; then
    litgpt download ${model2checkpoint[$model]} --checkpoint_dir $CHECKPOINTS_DIR --access_token $HF_TOKEN
    rm -rf $CHECKPOINTS_DIR/${model2checkpoint[$model]}/*.bin
fi
if [ ! -z ${model2checkpoint[${model}-instruct]} ]; then
    if [ ! -d $CHECKPOINTS_DIR/${model2checkpoint[${model}-instruct]} ]; then
        litgpt download ${model2checkpoint[${model}-instruct]} --checkpoint_dir $CHECKPOINTS_DIR --access_token $HF_TOKEN
        rm -rf $CHECKPOINTS_DIR/${model2checkpoint[${model}-instruct]}/*.bin
    fi
fi

# Datasets
declare -a DATASETS=(sst2 agnews dbpedia 20newsgroups banking77)

# Train sizes
declare -a FACTORS=(8 16 32 64 128 256 512)

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