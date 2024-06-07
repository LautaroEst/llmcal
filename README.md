# LLMs experiments

## Setup

Create the conda environment:
```bash
conda create -n llmcal python
conda activate llmcal
```

Run from the root of the repository (You will be asked to set up the checkpoints directory):
```bash
bash setup.sh
```

Reset terminal and run:
```bash
conda activate llmcal
bash scripts/run.00.download.sh
```

Run experiments for llama3:
```bash
bash scripts/llama3_all.sh
```