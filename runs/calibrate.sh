


SCRIPTS_DIR="./scripts"
RESULTS_DIR="./results"

MODEL="gpt2-xl"
declare -A DATASETS=(
    # ["glue/cola"]="2"
    ["glue/sst2"]="2"
    # ["glue/mrpc"]="2"
    # ["glue/mnli"]="3"
    # ["glue/qnli"]="2"
    # ["glue/rte"]="2"
)
TEMPLATE="0_shot"
TRAIN_SAMPLES=(50 200 800 2000 4000)
VALIDATION_SAMPLES=0
BASE_SEED=21738

for DATASET in "${!DATASETS[@]}"; do
    TRAIN_FILE=$RESULTS_DIR/run_dataset_on_model/$MODEL/$DATASET/train/$TEMPLATE
    EVAL_FILE=$RESULTS_DIR/run_dataset_on_model/$MODEL/$DATASET/validation/$TEMPLATE
    for NUM_SAMPLES in ${TRAIN_SAMPLES[@]}; do

        # Fine-tunned last layer (roofline)
        python $SCRIPTS_DIR/calibrate_features.py \
            --train_features $TRAIN_FILE/embeddings.npy \
            --train_labels $TRAIN_FILE/labels.npy \
            --eval_features $EVAL_FILE/embeddings.npy \
            --eval_labels $EVAL_FILE/labels.npy \
            --subsample_train $NUM_SAMPLES \
            --subsample_eval None \
            --validation_samples $VALIDATION_SAMPLES \
            --num_classes ${DATASETS[$DATASET]} \
            --method "affine" \
            --feature_map "None" \
            --alpha "matrix" \
            --bias \
            --loss "log-loss" \
            --batch_size "None" \
            --accelerator "cpu" \
            --num_devices 1 \
            --max_epochs 400 \
            --max_ls 40 \
            --tolerance 0.001 \
            --output_dir $RESULTS_DIR/calibrate_features/$MODEL/$DATASET/$TEMPLATE--$NUM_SAMPLES \
            --random_state $(expr $BASE_SEED + $NUM_SAMPLES)

        # Affine logits calibration
        for ALPHA in "matrix" "vector" "scalar" "None"; do
            python $SCRIPTS_DIR/calibrate_features.py \
                --train_features $TRAIN_FILE/logits.npy \
                --train_labels $TRAIN_FILE/labels.npy \
                --eval_features $EVAL_FILE/logits.npy \
                --eval_labels $EVAL_FILE/labels.npy \
                --subsample_train $NUM_SAMPLES \
                --subsample_eval None \
                --validation_samples $VALIDATION_SAMPLES \
                --num_classes ${DATASETS[$DATASET]} \
                --method "affine" \
                --feature_map "None" \
                --alpha $ALPHA \
                --bias \
                --loss "log-loss" \
                --batch_size "None" \
                --accelerator "cpu" \
                --num_devices 1 \
                --max_epochs 400 \
                --max_ls 40 \
                --tolerance 0.001 \
                --output_dir $RESULTS_DIR/calibrate_features/$MODEL/$DATASET/$TEMPLATE--$NUM_SAMPLES \
                --random_state $(expr $BASE_SEED + $NUM_SAMPLES)
        done

        # Affine embeddings calibration with feature map
        python $SCRIPTS_DIR/calibrate_features.py \
            --train_features $TRAIN_FILE/embeddings.npy \
            --train_labels $TRAIN_FILE/labels.npy \
            --eval_features $EVAL_FILE/embeddings.npy \
            --eval_labels $EVAL_FILE/labels.npy \
            --subsample_train $NUM_SAMPLES \
            --subsample_eval None \
            --validation_samples $VALIDATION_SAMPLES \
            --num_classes ${DATASETS[$DATASET]} \
            --method "affine" \
            --feature_map "quadratic" \
            --alpha "matrix" \
            --bias \
            --loss "log-loss" \
            --batch_size 64 \
            --accelerator "cpu" \
            --num_devices 1 \
            --max_epochs 200 \
            --max_ls 40 \
            --tolerance 0.001 \
            --output_dir $RESULTS_DIR/calibrate_features/$MODEL/$DATASET/$TEMPLATE--$NUM_SAMPLES \
            --random_state $(expr $BASE_SEED + $NUM_SAMPLES)

        # Mahalanobis embeddings calibration
        python $SCRIPTS_DIR/calibrate_features.py \
            --train_features $TRAIN_FILE/embeddings.npy \
            --train_labels $TRAIN_FILE/labels.npy \
            --eval_features $EVAL_FILE/embeddings.npy \
            --eval_labels $EVAL_FILE/labels.npy \
            --subsample_train $NUM_SAMPLES \
            --subsample_eval None \
            --validation_samples $VALIDATION_SAMPLES \
            --num_classes ${DATASETS[$DATASET]} \
            --method "mahalanobis" \
            --accelerator "cpu" \
            --num_devices 1 \
            --optimizer "Adam" \
            --batch_size 32 \
            --max_epochs 200 \
            --lr 0.001 \
            --weight_decay 0 \
            --tolerance 0.001 \
            --output_dir $RESULTS_DIR/calibrate_features/$MODEL/$DATASET/$TEMPLATE--$NUM_SAMPLES \
            --random_state $(expr $BASE_SEED + $NUM_SAMPLES)
        
        # Affine logits calibration with feature map
        python $SCRIPTS_DIR/calibrate_features.py \
            --train_features $TRAIN_FILE/logits.npy \
            --train_labels $TRAIN_FILE/labels.npy \
            --eval_features $EVAL_FILE/logits.npy \
            --eval_labels $EVAL_FILE/labels.npy \
            --subsample_train $NUM_SAMPLES \
            --subsample_eval None \
            --validation_samples $VALIDATION_SAMPLES \
            --num_classes ${DATASETS[$DATASET]} \
            --method "affine" \
            --feature_map "quadratic" \
            --alpha "matrix" \
            --bias \
            --loss "log-loss" \
            --batch_size "None" \
            --accelerator "cpu" \
            --num_devices 1 \
            --max_epochs 200 \
            --max_ls 40 \
            --tolerance 0.001 \
            --output_dir $RESULTS_DIR/calibrate_features/$MODEL/$DATASET/$TEMPLATE--$NUM_SAMPLES \
            --random_state $(expr $BASE_SEED + $NUM_SAMPLES)
        
        # Mahalanobis logits calibration
        python $SCRIPTS_DIR/calibrate_features.py \
            --train_features $TRAIN_FILE/logits.npy \
            --train_labels $TRAIN_FILE/labels.npy \
            --eval_features $EVAL_FILE/logits.npy \
            --eval_labels $EVAL_FILE/labels.npy \
            --subsample_train $NUM_SAMPLES \
            --subsample_eval None \
            --validation_samples $VALIDATION_SAMPLES \
            --num_classes ${DATASETS[$DATASET]} \
            --method "mahalanobis" \
            --accelerator "cpu" \
            --num_devices 1 \
            --optimizer "Adam" \
            --batch_size 32 \
            --max_epochs 200 \
            --lr 0.001 \
            --weight_decay 0 \
            --tolerance 0.001 \
            --output_dir $RESULTS_DIR/calibrate_features/$MODEL/$DATASET/$TEMPLATE--$NUM_SAMPLES \
            --random_state $(expr $BASE_SEED + $NUM_SAMPLES)

        # QDA logits calibration
        python $SCRIPTS_DIR/calibrate_features.py \
            --train_features $TRAIN_FILE/logits.npy \
            --train_labels $TRAIN_FILE/labels.npy \
            --eval_features $EVAL_FILE/logits.npy \
            --eval_labels $EVAL_FILE/labels.npy \
            --subsample_train $NUM_SAMPLES \
            --subsample_eval None \
            --validation_samples $VALIDATION_SAMPLES \
            --num_classes ${DATASETS[$DATASET]} \
            --method "qda" \
            --output_dir $RESULTS_DIR/calibrate_features/$MODEL/$DATASET/$TEMPLATE--$NUM_SAMPLES \
            --random_state $(expr $BASE_SEED + $NUM_SAMPLES)
        
        # LDA logits calibration
        python $SCRIPTS_DIR/calibrate_features.py \
            --train_features $TRAIN_FILE/logits.npy \
            --train_labels $TRAIN_FILE/labels.npy \
            --eval_features $EVAL_FILE/logits.npy \
            --eval_labels $EVAL_FILE/labels.npy \
            --subsample_train $NUM_SAMPLES \
            --subsample_eval None \
            --validation_samples $VALIDATION_SAMPLES \
            --num_classes ${DATASETS[$DATASET]} \
            --method "lda" \
            --output_dir $RESULTS_DIR/calibrate_features/$MODEL/$DATASET/$TEMPLATE--$NUM_SAMPLES \
            --random_state $(expr $BASE_SEED + $NUM_SAMPLES)

        # QDA embeddings calibration
        python $SCRIPTS_DIR/calibrate_features.py \
            --train_features $TRAIN_FILE/embeddings.npy \
            --train_labels $TRAIN_FILE/labels.npy \
            --eval_features $EVAL_FILE/embeddings.npy \
            --eval_labels $EVAL_FILE/labels.npy \
            --subsample_train $NUM_SAMPLES \
            --subsample_eval None \
            --validation_samples $VALIDATION_SAMPLES \
            --num_classes ${DATASETS[$DATASET]} \
            --method "qda" \
            --output_dir $RESULTS_DIR/calibrate_features/$MODEL/$DATASET/$TEMPLATE--$NUM_SAMPLES \
            --random_state $(expr $BASE_SEED + $NUM_SAMPLES)
        
        # LDA embeddings calibration
        python $SCRIPTS_DIR/calibrate_features.py \
            --train_features $TRAIN_FILE/embeddings.npy \
            --train_labels $TRAIN_FILE/labels.npy \
            --eval_features $EVAL_FILE/embeddings.npy \
            --eval_labels $EVAL_FILE/labels.npy \
            --subsample_train $NUM_SAMPLES \
            --subsample_eval None \
            --validation_samples $VALIDATION_SAMPLES \
            --num_classes ${DATASETS[$DATASET]} \
            --method "lda" \
            --output_dir $RESULTS_DIR/calibrate_features/$MODEL/$DATASET/$TEMPLATE--$NUM_SAMPLES \
            --random_state $(expr $BASE_SEED + $NUM_SAMPLES)
    done
done