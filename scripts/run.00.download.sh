

# Read token from token.txt
HF_TOKEN=$(cat hf_token.txt)

# repo_id=TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
repo_id=microsoft/Phi-3-mini-4k-instruct
# repo_id=meta-llama/Meta-Llama-3-8B

litgpt download $repo_id --checkpoint_dir $LIT_CHECKPOINTS --access_token $HF_TOKEN
