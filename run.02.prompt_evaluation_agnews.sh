#!/bin/bash

python scripts/main.py \
    --model tinyllama_3T_bf16 \
    --task tony_zhao_agnews_direct \
    --splits all
python scripts/main.py \
    --model affine_vector \
    --task tony_zhao_agnews_direct_tinyllama_3T_bf16_logits \
    --splits all \
    --model.num_classes 4
python scripts/view_results.py \
    --title "AG News - Direct prompt - TinyLlama (3T) - No adaptation" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_direct/tinyllama_3T_bf16/all
python scripts/view_results.py \
    --title "AG News - Direct prompt - TinyLlama (3T) - Affine Vector" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_direct_tinyllama_3T_bf16_logits/affine_vector_model.num_classes=4/all



python scripts/main.py \
    --model tinyllama_3T_bf16 \
    --task tony_zhao_agnews_orig \
    --splits all
python scripts/main.py \
    --model affine_vector \
    --task tony_zhao_agnews_orig_tinyllama_3T_bf16_logits \
    --splits all \
    --model.num_classes 4
python scripts/view_results.py \
    --title "AG News - Original prompt - TinyLlama (3T) - No adaptation" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_orig/tinyllama_3T_bf16/all
python scripts/view_results.py \
    --title "AG News - Original prompt - TinyLlama (3T) - Affine Vector" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_orig_tinyllama_3T_bf16_logits/affine_vector_model.num_classes=4/all


python scripts/main.py \
    --model tinyllama_3T_bf16 \
    --task tony_zhao_agnews_orig2 \
    --splits all
python scripts/main.py \
    --model affine_vector \
    --task tony_zhao_agnews_orig2_tinyllama_3T_bf16_logits \
    --splits all \
    --model.num_classes 4
python scripts/view_results.py \
    --title "AG News - Original prompt (V2) - TinyLlama (3T) - No adaptation" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_orig2/tinyllama_3T_bf16/all
python scripts/view_results.py \
    --title "AG News - Original prompt (V2) - TinyLlama (3T) - Affine Vector" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_orig2_tinyllama_3T_bf16_logits/affine_vector_model.num_classes=4/all


python scripts/main.py \
    --model tinyllama_3T_bf16 \
    --task tony_zhao_agnews_orig3 \
    --splits all
python scripts/main.py \
    --model affine_vector \
    --task tony_zhao_agnews_orig3_tinyllama_3T_bf16_logits \
    --splits all \
    --model.num_classes 4
python scripts/view_results.py \
    --title "AG News - Original prompt (V3) - TinyLlama (3T) - No adaptation" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_orig3/tinyllama_3T_bf16/all
python scripts/view_results.py \
    --title "AG News - Original prompt (V3) - TinyLlama (3T) - Affine Vector" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_orig3_tinyllama_3T_bf16_logits/affine_vector_model.num_classes=4/all


python scripts/main.py \
    --model tinyllama_3T_bf16 \
    --task tony_zhao_agnews_orig4 \
    --splits all
python scripts/main.py \
    --model affine_vector \
    --task tony_zhao_agnews_orig4_tinyllama_3T_bf16_logits \
    --splits all \
    --model.num_classes 4
python scripts/view_results.py \
    --title "AG News - Original prompt (V4) - TinyLlama (3T) - No adaptation" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_orig4/tinyllama_3T_bf16/all
python scripts/view_results.py \
    --title "AG News - Original prompt (V4) - TinyLlama (3T) - Affine Vector" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_orig4_tinyllama_3T_bf16_logits/affine_vector_model.num_classes=4/all


python scripts/main.py \
    --model tinyllama_3T_bf16 \
    --task tony_zhao_agnews_paper \
    --splits all
python scripts/main.py \
    --model affine_vector \
    --task tony_zhao_agnews_paper_tinyllama_3T_bf16_logits \
    --splits all \
    --model.num_classes 4
python scripts/view_results.py \
    --title "AG News - Paper prompt - TinyLlama (3T) - No adaptation" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_paper/tinyllama_3T_bf16/all
python scripts/view_results.py \
    --title "AG News - Paper prompt - TinyLlama (3T) - Affine Vector" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_paper_tinyllama_3T_bf16_logits/affine_vector_model.num_classes=4/all


python scripts/main.py \
    --model tinyllama_3T_bf16 \
    --task tony_zhao_agnews_inst \
    --splits all
python scripts/main.py \
    --model affine_vector \
    --task tony_zhao_agnews_inst_tinyllama_3T_bf16_logits \
    --splits all \
    --model.num_classes 4
python scripts/view_results.py \
    --title "AG News - Instruction following prompt - TinyLlama (3T) - No adaptation" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_inst/tinyllama_3T_bf16/all
python scripts/view_results.py \
    --title "AG News - Instruction following prompt - TinyLlama (3T) - Affine Vector" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_inst_tinyllama_3T_bf16_logits/affine_vector_model.num_classes=4/all


python scripts/main.py \
    --model tinyllama_3T_bf16 \
    --task tony_zhao_agnews_inst2 \
    --splits all
python scripts/main.py \
    --model affine_vector \
    --task tony_zhao_agnews_inst2_tinyllama_3T_bf16_logits \
    --splits all \
    --model.num_classes 4
python scripts/view_results.py \
    --title "AG News - Instruction following (V2) prompt - TinyLlama (3T) - No adaptation" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_inst2/tinyllama_3T_bf16/all
python scripts/view_results.py \
    --title "AG News - Instruction following (V2) prompt - TinyLlama (3T) - Affine Vector" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_inst2_tinyllama_3T_bf16_logits/affine_vector_model.num_classes=4/all


python scripts/main.py \
    --model tinyllama_3T_bf16 \
    --task tony_zhao_agnews_inst3 \
    --splits all
python scripts/main.py \
    --model affine_vector \
    --task tony_zhao_agnews_inst3_tinyllama_3T_bf16_logits \
    --splits all \
    --model.num_classes 4
python scripts/view_results.py \
    --title "AG News - Instruction following (V3) prompt - TinyLlama (3T) - No adaptation" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_inst3/tinyllama_3T_bf16/all
python scripts/view_results.py \
    --title "AG News - Instruction following (V3) prompt - TinyLlama (3T) - Affine Vector" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_inst3_tinyllama_3T_bf16_logits/affine_vector_model.num_classes=4/all


python scripts/main.py \
    --model tinyllama_3T_bf16 \
    --task tony_zhao_agnews_inst4 \
    --splits all
python scripts/main.py \
    --model affine_vector \
    --task tony_zhao_agnews_inst4_tinyllama_3T_bf16_logits \
    --splits all \
    --model.num_classes 4
python scripts/view_results.py \
    --title "AG News - Instruction following (V4) prompt - TinyLlama (3T) - No adaptation" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_inst4/tinyllama_3T_bf16/all
python scripts/view_results.py \
    --title "AG News - Instruction following (V4) prompt - TinyLlama (3T) - Affine Vector" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_inst4_tinyllama_3T_bf16_logits/affine_vector_model.num_classes=4/all


python scripts/main.py \
    --model tinyllama_3T_bf16 \
    --task tony_zhao_agnews_inst5 \
    --splits all
python scripts/main.py \
    --model affine_vector \
    --task tony_zhao_agnews_inst5_tinyllama_3T_bf16_logits \
    --splits all \
    --model.num_classes 4
python scripts/view_results.py \
    --title "AG News - Instruction following (V5) prompt - TinyLlama (3T) - No adaptation" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_inst5/tinyllama_3T_bf16/all
python scripts/view_results.py \
    --title "AG News - Instruction following (V5) prompt - TinyLlama (3T) - Affine Vector" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_inst5_tinyllama_3T_bf16_logits/affine_vector_model.num_classes=4/all


python scripts/main.py \
    --model tinyllama_3T_bf16 \
    --task tony_zhao_agnews_inst6 \
    --splits all
python scripts/main.py \
    --model affine_vector \
    --task tony_zhao_agnews_inst6_tinyllama_3T_bf16_logits \
    --splits all \
    --model.num_classes 4
python scripts/view_results.py \
    --title "AG News - Instruction following (V6) prompt - TinyLlama (3T) - No adaptation" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_inst6/tinyllama_3T_bf16/all
python scripts/view_results.py \
    --title "AG News - Instruction following (V6) prompt - TinyLlama (3T) - Affine Vector" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_inst6_tinyllama_3T_bf16_logits/affine_vector_model.num_classes=4/all


python scripts/main.py \
    --model tinyllama_3T_bf16 \
    --task tony_zhao_agnews_inst7 \
    --splits all
python scripts/main.py \
    --model affine_vector \
    --task tony_zhao_agnews_inst7_tinyllama_3T_bf16_logits \
    --splits all \
    --model.num_classes 4
python scripts/view_results.py \
    --title "AG News - Instruction following (V7) prompt - TinyLlama (3T) - No adaptation" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_inst7/tinyllama_3T_bf16/all
python scripts/view_results.py \
    --title "AG News - Instruction following (V7) prompt - TinyLlama (3T) - Affine Vector" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_inst7_tinyllama_3T_bf16_logits/affine_vector_model.num_classes=4/all


python scripts/main.py \
    --model tinyllama_3T_bf16 \
    --task tony_zhao_agnews_inst8 \
    --splits all
python scripts/main.py \
    --model affine_vector \
    --task tony_zhao_agnews_inst8_tinyllama_3T_bf16_logits \
    --splits all \
    --model.num_classes 4
python scripts/view_results.py \
    --title "AG News - Instruction following (V8) prompt - TinyLlama (3T) - No adaptation" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_inst8/tinyllama_3T_bf16/all
python scripts/view_results.py \
    --title "AG News - Instruction following (V8) prompt - TinyLlama (3T) - Affine Vector" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_inst8_tinyllama_3T_bf16_logits/affine_vector_model.num_classes=4/all


python scripts/main.py \
    --model tinyllama_3T_bf16 \
    --task tony_zhao_agnews_mc \
    --splits all
python scripts/main.py \
    --model affine_vector \
    --task tony_zhao_agnews_mc_tinyllama_3T_bf16_logits \
    --splits all \
    --model.num_classes 4
python scripts/view_results.py \
    --title "AG News - Multiple Choice prompt - TinyLlama (3T) - No adaptation" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_mc/tinyllama_3T_bf16/all
python scripts/view_results.py \
    --title "AG News - Multiple Choice prompt - TinyLlama (3T) - Affine Vector" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_mc_tinyllama_3T_bf16_logits/affine_vector_model.num_classes=4/all


python scripts/main.py \
    --model tinyllama_3T_bf16 \
    --task tony_zhao_agnews_mcw \
    --splits all
python scripts/main.py \
    --model affine_vector \
    --task tony_zhao_agnews_mcw_tinyllama_3T_bf16_logits \
    --splits all \
    --model.num_classes 4
python scripts/view_results.py \
    --title "AG News - Multple Choice with words prompt - TinyLlama (3T) - No adaptation" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_mcw/tinyllama_3T_bf16/all
python scripts/view_results.py \
    --title "AG News - Multple Choice with words prompt - TinyLlama (3T) - Affine Vector" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_mcw_tinyllama_3T_bf16_logits/affine_vector_model.num_classes=4/all

python scripts/main.py \
    --model tinyllama_3T_bf16 \
    --task tony_zhao_agnews_nli \
    --splits all
python scripts/main.py \
    --model affine_vector \
    --task tony_zhao_agnews_nli_tinyllama_3T_bf16_logits \
    --splits all \
    --model.num_classes 4
python scripts/view_results.py \
    --title "AG News - NLI prompt - TinyLlama (3T) - No adaptation" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_nli/tinyllama_3T_bf16/all
python scripts/view_results.py \
    --title "AG News - NLI prompt - TinyLlama (3T) - Affine Vector" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_nli_tinyllama_3T_bf16_logits/affine_vector_model.num_classes=4/all

python scripts/main.py \
    --model tinyllama_3T_bf16 \
    --task tony_zhao_agnews_qa \
    --splits all
python scripts/main.py \
    --model affine_vector \
    --task tony_zhao_agnews_qa_tinyllama_3T_bf16_logits \
    --splits all \
    --model.num_classes 4
python scripts/view_results.py \
    --title "AG News - Question Answering prompt - TinyLlama (3T) - No adaptation" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_qa/tinyllama_3T_bf16/all
python scripts/view_results.py \
    --title "AG News - Question Answering prompt - TinyLlama (3T) - Affine Vector" \
    --metrics norm_cross_entropy,accuracy,f1_score \
    --bootstrap 100 \
    --random_state 9287 \
    --test False \
    tony_zhao_agnews_qa_tinyllama_3T_bf16_logits/affine_vector_model.num_classes=4/all