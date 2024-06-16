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

Reset terminal. Write your HF token on a file named `hf_token.txt` and then run:
```bash
conda activate llmcal
bash scripts/run.00.download.sh
```

Run experiments for tinyllama:
```bash
bash scripts/all.sh
```

You can inspect the results in the `results.ipynb` and `timing.ipynb` notebooks.