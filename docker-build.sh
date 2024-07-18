#!/usr/bin/env bash

SCRIPT_PATH="$(dirname $0)"
cd $SCRIPT_PATH

docker build -f Dockerfile -t europe-west4-docker.pkg.dev/machine-learning-429020/musicnet/musicnet ./