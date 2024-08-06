#!/bin/bash -e

for dataset in ./experiments.llama3_all/*; do
    for prompt in $dataset/*; do
        for model in $prompt/*; do
            for base_method in $model/*; do
                for cal_method in $base_method/*; do
                    for split in train validation test; do
                        if [ -d $base_method/.cache/predictions/$split ] && [ ! -e $cal_method/predictions/$split ]; then
                            ln -s ../../.cache/predictions/$split $cal_method/predictions/$split
                            echo $cal_method/predictions/$split
                        fi
                    done
                done
            done
        done
    done
done