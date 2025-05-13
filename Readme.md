# Large Language Models adaptation using post-hoc calibration (PHC) and supervised fine-tuning (SFT)

This repo contains the code to run adaptation of generative language models. The `scripts` directory contains the scripts to be run and everything (datasets, number of adaptation samples, base model) can be configured using the `scripts/env.sh` file. 

## Adaptation methods

Supported adaptation methods are distributed across different scripts files:
- `scripts/cal_vs_samples.sh`: Compute the logits for the current model and then run Post-hoc calibration (PHC) with affine calibration for four different variants:
  - Vector Scaling (**PHC-VS**): The matrix of the affine transformation is assumed to be diagonal.
  - Direction Preserving (**PHC-DP**): The matrix of the affine transformation is replaced by a scalar, $\alpha$.
  - Bias Only (**PHC-BO**): Same as PHC-DP, but with the weight $\alpha$ fixed at 1.
  - Temperature Scaling (**PHC-TS**): Same as PHC-DP but with the bias term fixed at 0.
- `scripts/lora_vs_samples.sh`: Run SFT using LoRA on all the datasets. There are three variants implemented depending on how we use the adaptation samples:
  - **SFT-w-val**: we leave a portion of the adaptation set for validation, and stop training based on the CE loss on this set.
  - **SFT-retrain**: we perform SFT with validation, as above, and keep track of the optimal number of steps. Then, we rerun SFT on the full training set, stopping after that same number of steps. 
  - **SFT-wo-val**: we use all the available samples for training until convergence based on the training loss.
SFT and computing the model posteriors is implemented using the [litgpt](https://github.com/Lightning-AI/litgpt) library.

## Datasets and Models

Currently, there are 5 supported datasets:
- **SST-2**: The Stanford Sentiment Treebank, which includes movie reviews annotated as either positive or negative.
- **AGNews**: a collection of news articles grouped into four categories. 
- **DBPedia**: a dataset derived from Wikipedia, where each article is categorized into one of 14 topics.
- **20NewsGroups**: posts from online newsgroups categorized into 20 different topics. 
- **Banking77**: customer service queries related to online banking, classified into 77 intent categories. 

All the models of the `litgpt>=0.5.3` library are supported, but for our experiments we used Llama3.2-1B and Qwen2.5-7B. Be sure to configure your token in order to use model that require permission.

## Installation

To use this you can clone it and pip-install it direclty but we recommend creating a separate conda enviroment:
```bash
conda create -n llmcal python=3.11
conda activate llmcal
pip install -r requirements.txt
pip install -e .
```

## Cite

Coming soon in OpenReview!
