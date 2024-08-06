from huggingface_hub import snapshot_download
from pathlib import Path

def main(repo_id, checkpoints_dir, access_token = None):
    directory = Path(checkpoints_dir) / repo_id
    download_files = ["tokenizer*", "config.json", "*.bin*"]
    snapshot_download(
        repo_id,
        local_dir=directory,
        allow_patterns=download_files,
        token=access_token,
    )

if __name__ == "__main__":
    from fire import Fire
    Fire(main)
