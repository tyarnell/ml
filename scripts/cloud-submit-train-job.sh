#!/bin/bash

echo "Submitting a Cloud ML Engine job..."

BUCKET="train-jobs" # change to your bucket name
MODEL_NAME="isicTrainer" # change to your model name
CURRENT_DATE=`date +%Y%m%d_%H%M%S`

JOB_NAME=tune_${MODEL_NAME}_${CURRENT_DATE}
TRAINER_PACKAGE_PATH=/Users/Tyler/projects/ml/xgb_trainer # this can be a gcs location to a zipped and uploaded package
MAIN_TRAINER_MODULE=xgb_trainer.task
JOB_DIR=gs://train-jobs/melanoma/job
CONFIG=/Users/Tyler/projects/ml/machine_configs/hptuning_config.yaml
REGION=us-west1
RUNTIME_VERSION=1.12
DATA_DIR=gs://${BUCKET}/melanoma/data

gcloud ml-engine jobs submit training $JOB_NAME \
 --package-path $TRAINER_PACKAGE_PATH \
 --module-name $MAIN_TRAINER_MODULE \
 --job-dir $JOB_DIR \
 --config $CONFIG \
 --region $REGION \
 --runtime-version $RUNTIME_VERSION \
 -- \
 --data-dir $DATA_DIR