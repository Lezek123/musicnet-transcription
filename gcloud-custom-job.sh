#!/usr/bin/env bash

set -x

SCRIPT_PATH="$(dirname $0)"
cd $SCRIPT_PATH

CONTAINER_IMAGE_URI="europe-west4-docker.pkg.dev/machine-learning-429020/musicnet/musicnet"
BUILD_LOCATION="europe-west1"
JOB_LOCATION="europe-west4"
JOB_NAME="Test job 1"
# https://cloud.google.com/vertex-ai/pricing#training
# https://cloud.google.com/compute/docs/general-purpose-machines#n1_machines
# This one has 8 vCPUs, 52 GB of RAM and costs $0.6 / hr (in eu-west4)
# We can cache the entire dataset in RAM which may speed up the training and allow full shuffling
MACHINE_TYPE="n1-highmem-8"
# https://cloud.google.com/vertex-ai/pricing#training
# NVidia Tesla T4 costs $0.4025 / hr and has 16 GB of memory
ACCELERATOR_TYPE="NVIDIA_TESLA_T4"
ACCELERATOR_COUNT=1
REPLICA_COUNT=1
# For disk spec we'll use the default 100 GB PD-SSD which costs a negible amount ($0.03 / hr)
# https://cloud.google.com/vertex-ai/docs/reference/rest/v1beta1/DiskSpec

# 1. Build the image on Google Cloud
# Build costs TODO: per hour
# Storing ~10 GB image costs ~$0,5 per month
# https://cloud.google.com/compute/disks-image-pricing#imagestorage
# gcloud builds submit \
#     --region=$BUILD_LOCATION \
#     --tag $CONTAINER_IMAGE_URI

# 2. Create the custom job
gcloud ai custom-jobs create \
  --region=$JOB_LOCATION \
  --display-name="$JOB_NAME" \
  --worker-pool-spec=machine-type=$MACHINE_TYPE,accelerator-type=$ACCELERATOR_TYPE,accelerator-count=$ACCELERATOR_COUNT,replica-count=$REPLICA_COUNT,container-image-uri=$CONTAINER_IMAGE_URI