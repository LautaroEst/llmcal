


SCRIPTS_DIR="./scripts"
RESULTS_DIR="./results"

MODEL="gpt2-xl"


DATASET="tony_zhao/sst2"
TEMPLATE="0_shot"
TRAIN_FILE=$RESULTS_DIR/run_dataset_on_model/$MODEL/$DATASET/train/$TEMPLATE
EVAL_FILE=$RESULTS_DIR/run_dataset_on_model/$MODEL/$DATASET/test/$TEMPLATE

for NUM_SAMPLES in 50 100 200 400 800 1000; do
    python $SCRIPTS_DIR/calibrate_features.py \
        --train_features $TRAIN_FILE/logits.npy \
        --train_labels $TRAIN_FILE/labels.npy \
        --eval_features $EVAL_FILE/logits.npy \
        --eval_labels $EVAL_FILE/labels.npy \
        --subsample_train $NUM_SAMPLES \
        --subsample_eval None \
        --num_classes 2 \
        --method "affine" \
        --alpha "vector" \
        --bias \
        --loss "log-loss" \
        --accelerator "cpu" \
        --num_devices 1 \
        --batch_size 32 \
        --num_epochs 1000 \
        --lr 0.001 \
        --weight_decay 0.0 \
        --tolerance 1e-4 \
        --output_dir $RESULTS_DIR/calibrate_features/$MODEL/$DATASET/$TEMPLATE--$NUM_SAMPLES \
        --random_state 83629
done