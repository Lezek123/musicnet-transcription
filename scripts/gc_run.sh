#!/usr/bin/env bash

SCRIPT_PATH="$(dirname $0)"
cd $SCRIPT_PATH/..

dvc remote add origin -d /gcs/musicnet-dvc
./scripts/run_pipeline.sh exp=True
dvc exp push origin