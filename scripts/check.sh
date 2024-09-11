
# exp_dir=experiments.llama3.09-09-2024
# exp_dir=experiments
exp_dir=experiments.tinyllama.ok.11-09-2024
for dataset in 20newsgroups agnews banking77 dbpedia sst2; do
    prompt="encoder_${dataset}"
    # model="lm_tinyllama"
    # model="lm_phi3"
    model="lm_llama3"
    for size in "_1_" "_2_" "_4_" "_8_" "_16_" "_32_" "_64_" "_128_" "_256_" "_512_"; do
        # echo $(find  experiments/ -name "test" -path "experiments/${dataset}_*" ! -path "*xval*" ! -path "*/.cache/*" ! -path "*_no_es*" ! -path "*_train_on_val*" | grep $size)
        counts=$(find  $exp_dir/ -name "test" -path "$exp_dir/${dataset}_*" ! -path "*xval*" ! -path "*/.cache/*" ! -path "*_no_es*" ! -path "*.v*" | grep $size | wc -l)
        if [ $counts -eq 0 ]; then
            continue
        fi
        echo "Dataset: $dataset, Size: $size, Counts: $counts"
        # find  experiments/ -name "test" -path "experiments/${dataset}_*" ! -path "*xval*" ! -path "*/.cache/*" ! -path "*_no_es*" | grep $size
        # echo
    done
    echo
done



