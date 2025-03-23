CHECKPOINTS_DIR=../llmcal2/outputs/checkpoints
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
# num_seeds=9

# Supported models
declare -A model2checkpoint=(
    ["llama3.2-1b"]="meta-llama/Llama-3.2-1B"
    ["llama3.2-1b-instruct"]="meta-llama/Llama-3.2-1B-Instruct"
    ["qwen2.5-7b"]="Qwen/Qwen2.5-7B"
    ["qwen2.5-7b-instruct"]="Qwen/Qwen2.5-7B-Instruct"
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
declare -a DATASETS=(20newsgroups dbpedia sst2 agnews banking77)
# declare -a DATASETS=(sst2 agnews)

# Train sizes
declare -a FACTORS=(8 16 32 64 128 256)

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