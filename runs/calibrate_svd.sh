


SCRIPTS_DIR="./scripts"
RESULTS_DIR="./results"

MODEL="gpt2-xl"
declare -A DATASETS=(
    # ["glue/cola"]="2"
    ["glue/sst2"]="2"
    ["glue/mrpc"]="2"
    ["glue/mnli"]="3"
    # ["glue/qnli"]="2"
    # ["glue/rte"]="2"
    # ["glue/wnli"]="2"
)
TEMPLATE="0_shot"
declare -A TRAIN_SAMPLES=(
    ["glue/cola"]="50 200 800 2000 3800"
    ["glue/sst2"]="50 200 800 2000 3800"
    ["glue/mrpc"]="50 200 800 2000 3468"
    ["glue/mnli"]="50 200 800 2000 3800"
    ["glue/qnli"]="50 200 800 2000 3800"
    ["glue/rte"]="50 200 800 2290"
    ["glue/wnli"]="50 200 435"
)
BASE_SEED=21738
VALIDATION_SAMPLES=200

DEVICE="gpu"

for DATASET in "${!DATASETS[@]}"; do
    TRAIN_FILE=$RESULTS_DIR/run_dataset_on_model/$MODEL/$DATASET/train/$TEMPLATE
    EVAL_FILE=$RESULTS_DIR/run_dataset_on_model/$MODEL/$DATASET/validation/$TEMPLATE
    IFS=" " read -r -a TRAIN_SAMPLES_DATASET <<< ${TRAIN_SAMPLES[$DATASET]}

    echo "========================================"
    echo Model: $MODEL
    echo Dataset: $DATASET
    echo "Training samples: ${TRAIN_SAMPLES_DATASET[@]}"
    echo "========================================"
    echo ""

    for NUM_SAMPLES in ${TRAIN_SAMPLES_DATASET[@]}; do

        # QR Mahalanobis embeddings calibration
        python $SCRIPTS_DIR/calibrate_features.py \
            --train_features $TRAIN_FILE/embeddings.npy \
            --train_labels $TRAIN_FILE/labels.npy \
            --eval_features $EVAL_FILE/embeddings.npy \
            --eval_labels $EVAL_FILE/labels.npy \
            --subsample_train $NUM_SAMPLES \
            --subsample_eval None \
            --validation_samples $VALIDATION_SAMPLES \
            --num_classes ${DATASETS[$DATASET]} \
            --method "mahalanobis_qr" \
            --feature_map "identity" \
            --accelerator $DEVICE \
            --devices 1 \
            --optimizer "Adam" \
            --batch_size 32 \
            --max_epochs 400 \
            --learning_rate 0.00001 \
            --weight_decay 0 \
            --patience 1000000 \
            --output_dir $RESULTS_DIR/calibrate_features/$MODEL/$DATASET/$TEMPLATE--$NUM_SAMPLES \
            --random_state $(expr $BASE_SEED + $NUM_SAMPLES)
            
        # # QR Mahalanobis logits calibration
        # python $SCRIPTS_DIR/calibrate_features.py \
        #     --train_features $TRAIN_FILE/logits.npy \
        #     --train_labels $TRAIN_FILE/labels.npy \
        #     --eval_features $EVAL_FILE/logits.npy \
        #     --eval_labels $EVAL_FILE/labels.npy \
        #     --subsample_train $NUM_SAMPLES \
        #     --subsample_eval None \
        #     --validation_samples $VALIDATION_SAMPLES \
        #     --num_classes ${DATASETS[$DATASET]} \
        #     --method "mahalanobis_svd" \
        #     --feature_map "identity" \
        #     --accelerator $DEVICE \
        #     --devices 1 \
        #     --optimizer "Adam" \
        #     --batch_size $NUM_SAMPLES \
        #     --max_epochs 100 \
        #     --learning_rate 0.00001 \
        #     --weight_decay 0 \
        #     --patience 1000000 \
        #     --output_dir $RESULTS_DIR/calibrate_features/$MODEL/$DATASET/$TEMPLATE--$NUM_SAMPLES \
        #     --random_state $(expr $BASE_SEED + $NUM_SAMPLES)
    done
done