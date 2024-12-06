#!/bin/bash -ex

run_no_adaptation() {
    echo running no adaptation $1 $2 $3
}


main() {
    model=$1
    for dataset in "${!dataset2trainsizes[@]}"; do
        for size in ${dataset2trainsizes[$dataset]}; do
            for num_seed in $(seq 0 $((num_seeds-1))); do
                # Baseline
                run_no_adaptation $dataset $size $num_seed
            done
        done
    done
}



