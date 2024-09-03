#!/usr/bin/env bash

set -x

SCRIPT_PATH="$(dirname $0)"
cd $SCRIPT_PATH

COMMAND=$1
CONTAINER_IMAGE_URI="europe-west4-docker.pkg.dev/machine-learning-429020/musicnet/musicnet"
BUILD_LOCATION="europe-west1"
JOB_LOCATION="europe-west4"

if [[ "$COMMAND" == "build" ]]; then
  # 1. Build the image on Google Cloud
  # Build costs TODO: per hour
  # Storing ~10 GB image costs ~$0,5 per month
  # https://cloud.google.com/compute/disks-image-pricing#imagestorage
  gcloud builds submit \
      --region=$BUILD_LOCATION \
      --tag $CONTAINER_IMAGE_URI
elif [[ "$COMMAND" == "create" ]]; then
  JOB_NAME=$2
  if [ -z "$JOB_NAME" ]; then
    echo "Missing job name"
    exit 1
  fi
  # 2. Create the custom job
  gcloud ai custom-jobs create \
    --region=$JOB_LOCATION \
    --display-name="$JOB_NAME" \
    --config ./gc-job.yaml
else
  echo "Command unknown or not provided."
fi