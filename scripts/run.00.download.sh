

# Read token from token.txt
HF_TOKEN=$(cat hf_token.txt)

litgpt download --repo_id TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --checkpoint_dir $LIT_CHECKPOINTS
