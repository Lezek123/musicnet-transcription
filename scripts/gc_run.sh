#!/usr/bin/env bash

SCRIPT_PATH="$(dirname $0)"
cd $SCRIPT_PATH/..

dvc remote add origin /gcs/musicnet-dvc
./scripts/run_pipeline.sh stages=[GENERATE,PREPROCESS,TRAIN,EVAL] exp=True
dvc exp push origin