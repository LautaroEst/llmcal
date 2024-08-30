python -m llmcal dbpedia_2_9722 basic_dbpedia_0-shot_litgpt lm_phi3 lora_20samples no_calibration --accelerator "gpu"
python -m llmcal dbpedia_128_131 basic_dbpedia_0-shot_litgpt lm_phi3 lora_1000samples no_calibration --accelerator "gpu"

python -m llmcal 20newsgroups_2_927 basic_20newsgroups_0-shot_litgpt lm_phi3 lora_40samples no_calibration --accelerator "gpu"
python -m llmcal 20newsgroups_128_129 basic_20newsgroups_0-shot_litgpt lm_phi3 lora_1000samples no_calibration --accelerator "gpu"

python -m llmcal 20newsgroups_2_435 basic_20newsgroups_0-shot_litgpt lm_phi3 lora_40samples no_calibration --accelerator "gpu"
python -m llmcal 20newsgroups_128_131 basic_20newsgroups_0-shot_litgpt lm_phi3 lora_1000samples no_calibration --accelerator "gpu"

python -m llmcal 20newsgroups_2_972 basic_20newsgroups_0-shot_litgpt lm_phi3 lora_40samples no_calibration --accelerator "gpu"
python -m llmcal 20newsgroups_128_543 basic_20newsgroups_0-shot_litgpt lm_phi3 lora_1000samples no_calibration --accelerator "gpu"

python -m llmcal 20newsgroups_2_4351 basic_20newsgroups_0-shot_litgpt lm_phi3 lora_40samples no_calibration --accelerator "gpu"
python -m llmcal 20newsgroups_128_878 basic_20newsgroups_0-shot_litgpt lm_phi3 lora_1000samples no_calibration --accelerator "gpu"

python -m llmcal 20newsgroups_2_9722 basic_20newsgroups_0-shot_litgpt lm_phi3 lora_40samples no_calibration --accelerator "gpu"
python -m llmcal 20newsgroups_128_909 basic_20newsgroups_0-shot_litgpt lm_phi3 lora_1000samples no_calibration --accelerator "gpu"