#!/usr/bin/env bash

SCRIPT_PATH="$(dirname $0)"
cd $SCRIPT_PATH
SCRIPT_PATH=$(pwd)

set -x

# TODO: Setup a remote for DVC

STAGES="$1"
C_FLAG="$2"

if [ $C_FLAG == "-c" ]; then
    FLAGS="-c job"
else
    FLAGS=""
fi

python3 -m musicnet.main $FLAGS\
        --config-path="$SCRIPT_PATH" \
        --config-name=defaults \
        stages=[$STAGES] \
        # Can add some overrides here, like:
        # model.n_neurons=200 \