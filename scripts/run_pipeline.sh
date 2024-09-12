#!/usr/bin/env bash

SCRIPT_PATH="$(dirname $0)"
cd $SCRIPT_PATH
SCRIPT_PATH=$(pwd)

set -x

set -a
. ../.env
set +a

python3 -m musicnet.main \
        --config-path="$SCRIPT_PATH" \
        --config-name=defaults \
        $@
        # Can add some overrides here, like:
        # model.n_neurons=200 \