#!/usr/bin/env bash

set -x

SCRIPT_PATH="$(dirname $0)"
cd $SCRIPT_PATH

CONTAINER_IMAGE_URI="europe-west4-docker.pkg.dev/machine-learning-429020/musicnet/musicnet"
BUILD_LOCATION="europe-west1"
JOB_LOCATION="europe-west4"
JOB_NAME="Test job 1"
MACHINE_TYPE="g2-standard-8"
REPLICA_COUNT=1

# 1. Build the image on Google Cloud
# gcloud builds submit \
#     --region=$BUILD_LOCATION \
#     --tag $CONTAINER_IMAGE_URI

# 2. Create the custom job
gcloud ai custom-jobs create \
  --region=$JOB_LOCATION \
  --display-name="$JOB_NAME" \
  --worker-pool-spec=machine-type=$MACHINE_TYPE,accelerator-type=NVIDIA_L4,replica-count=$REPLICA_COUNT,container-image-uri=$CONTAINER_IMAGE_URI