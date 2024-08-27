#!/usr/bin/env bash

SCRIPT_PATH="$(dirname $0)"
cd $SCRIPT_PATH
SCRIPT_PATH=$(pwd)

set -x

# TODO: Setup a remote for DVC

python3 -m musicnet.main $FLAGS\
        --config-path="$SCRIPT_PATH" \
        --config-name=defaults \
        $@
        # Can add some overrides here, like:
        # model.n_neurons=200 \