python -m llmcal dbpedia_2_9722 basic_dbpedia_0-shot_litgpt lm_llama3 lora_20samples no_calibration --accelerator "gpu"

python -m llmcal 20newsgroups_2_927 basic_20newsgroups_0-shot_litgpt lm_llama3 lora_40samples no_calibration --accelerator "gpu"
python -m llmcal 20newsgroups_2_435 basic_20newsgroups_0-shot_litgpt lm_llama3 lora_40samples no_calibration --accelerator "gpu"
python -m llmcal 20newsgroups_2_972 basic_20newsgroups_0-shot_litgpt lm_llama3 lora_40samples no_calibration --accelerator "gpu"
python -m llmcal 20newsgroups_2_4351 basic_20newsgroups_0-shot_litgpt lm_llama3 lora_40samples no_calibration --accelerator "gpu"
python -m llmcal 20newsgroups_2_9722 basic_20newsgroups_0-shot_litgpt lm_llama3 lora_40samples no_calibration --accelerator "gpu"

bash scripts/lm_llama3-1/20newsgroups.sh
bash scripts/lm_llama3-2/20newsgroups.sh
bash scripts/lm_llama3-3/20newsgroups.sh
bash scripts/lm_llama3-4/20newsgroups.sh
bash scripts/lm_llama3-5/20newsgroups.sh
