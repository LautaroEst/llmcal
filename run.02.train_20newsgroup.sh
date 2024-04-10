#!/bin/bash

# python scripts/main.py \
#     --model tinyllama_3T_bf16 \
#     --task 20newsgroup \
#     --splits all
python scripts/view_results.py \
    --title "20 newsgroup - Instruction following prompt - TinyLlama (3T) - No adaptation" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test True \
    20newsgroup/tinyllama_3T_bf16/all

# python scripts/main.py \
#     --model tinyllama_3T_bf16_lora \
#     --task 20newsgroup \
#     --splits n=1000_rs=8389 \
#     --train.learning_rate 0.0005 \
#     --train.max_epochs 20 \
#     --model.lora_r 8
python scripts/view_results.py \
    --title "20 newsgroup - Instruction following prompt - TinyLlama (3T) - Lora (n = 1000)" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test True \
    20newsgroup/tinyllama_3T_bf16_lora_train.learning_rate=0.0005_train.max_epochs=20_model.lora_r=8/n=1000_rs=8389

# python scripts/main.py \
#     --model tinyllama_3T_bf16_lora \
#     --task 20newsgroup \
#     --splits n=100_rs=7384 \
#     --train.learning_rate 0.0005 \
#     --train.max_epochs 40 \
#     --model.lora_r 8
# python scripts/view_results.py \
#     --title "20 newsgroup - Instruction following prompt - TinyLlama (3T) - Lora (n = 100)" \
#     --metrics norm_cross_entropy,accuracy,f1_score \
#     --bootstrap 100 \
#     --random_state 9287 \
#     --test True \
#     20newsgroup/tinyllama_3T_bf16_lora_train.learning_rate=0.0005_train.max_epochs=20_model.lora_r=8/n=100_rs=7384

python scripts/main.py \
    --model affine_vector \
    --task 20newsgroup_tinyllama_3T_bf16_logits \
    --splits n=1000_rs=8389 \
    --model.num_classes 20
python scripts/view_results.py \
    --title "20 newsgroup - Instruction following prompt - TinyLlama (3T) - Affine Vector (n = 1000)" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test True \
    20newsgroup_tinyllama_3T_bf16_logits/affine_vector_model.num_classes=20/n=1000_rs=8389


python scripts/main.py \
    --model affine_vector \
    --task 20newsgroup_tinyllama_3T_bf16_logits \
    --splits n=100_rs=7384 \
    --model.num_classes 20 \
    --train.max_epochs 2
python scripts/view_results.py \
    --title "20 newsgroup - Instruction following prompt - TinyLlama (3T) - Affine Vector (n = 100)" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test True \
    20newsgroup_tinyllama_3T_bf16_logits/affine_vector_model.num_classes=20_train.max_epochs=10/n=100_rs=7384


