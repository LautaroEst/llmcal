
for dataset in 20newsgroups agnews banking77 dbpedia sst2; do
    prompt="basic_${dataset}_0-shot_litgpt"
    model="lm_tinyllama"
    for size in "_2_" "_4_" "_8_" "_16_" "_32_" "_64_" "_256_"; do
        counts=$(find  experiments/ -name "test" -path "experiments/${dataset}_*" ! -path "*xval*" ! -path "*/.cache/*" ! -path "*_no_es*" | grep $size | wc -l)
        if [ $counts -eq 0 ]; then
            continue
        fi
        echo "Dataset: $dataset, Size: $size, Counts: $counts"
        # find  experiments/ -name "test" -path "experiments/${dataset}_*" ! -path "*xval*" ! -path "*/.cache/*" ! -path "*_no_es*" | grep $size
        # echo
    done
    echo
done



