#!/usr/bin/env bash

SCRIPT_PATH="$(dirname $0)"
cd $SCRIPT_PATH
SCRIPT_PATH=$(pwd)

set -x

# 1. Run pipeline using Hydra instead of DVC
# 2. Log params using DVCLive
# 3. Commit experiments
# 4. Setup a remote for DVC

# Prepare the dataset
python3 -m musicnet.preprocessing.generate \
    --config-path="$SCRIPT_PATH" \
    --config-name=test_cfg

python3 -m musicnet.preprocessing.preprocess \
    --config-path="$SCRIPT_PATH" \
    --config-name=test_cfg

python3 -m musicnet.models.train \
    --config-path="$SCRIPT_PATH" \
    --config-name=test_cfg \
    model.n_neurons=200 \
    model.kernel_size=3 \
    model.n_layers=5 \
    model.epochs=20
    model.activation=GELU