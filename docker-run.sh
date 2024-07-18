#!/usr/bin/env bash

set -x

SCRIPT_PATH="$(dirname $0)"
cd $SCRIPT_PATH

DS_DIR="$(pwd)/data/MusicNet"

if [ -z "$WITH_BASH" ]; then
    FLAGS=""
else
    FLAGS="-it --entrypoint /bin/bash"
fi

docker run \
    $FLAGS \
    -v $DS_DIR:/musicnet/data/MusicNet \
    --gpus all \
    europe-west4-docker.pkg.dev/machine-learning-429020/musicnet/musicnet