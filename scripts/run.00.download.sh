
conda activate llmcal

# Read token from token.txt
HF_TOKEN=$(cat hf_token.txt)

# litgpt download --repo_id TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --checkpoint_dir $LIT_CHECKPOINTS
# # litgpt download --repo_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --checkpoint_dir $LIT_CHECKPOINTS
# litgpt download --repo_id microsoft/phi-2 --checkpoint_dir $LIT_CHECKPOINTS
litgpt download --repo_id meta-llama/Meta-Llama-3-8B --checkpoint_dir $LIT_CHECKPOINTS --access_token $HF_TOKEN