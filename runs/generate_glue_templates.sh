

SEEDS=(28347 102 73901 7384 8275)
DATASETS=("glue/sst2" )

for DATASET in ${DATASETS[@]}; do
    for SEED in ${SEEDS[@]}; do
        python scripts/generate_template.py \
            --dataset_name ${DATASET} \
            --output_dir runs/templates \
            --num_shots "0,1,4,8" \
            --seed ${SEED}
    done 
done