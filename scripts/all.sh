#!/bin/bash -ex

./scripts/prepare_data.sh
./scripts/cal_vs_samples.sh
./scripts/lora_vs_samples.sh
./scripts/lora_plus_vs_samples.sh