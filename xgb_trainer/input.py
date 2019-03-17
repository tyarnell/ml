import sys
import os

import json
import multiprocessing

import pandas as pd

import xgboost as xgb
from google.cloud import storage
from sklearn.preprocessing import MinMaxScaler

from xgb_trainer import metadata

def split_gcs_path(gcs_path):
    '''
    Retrieve the bucket name for a GCS bucket from a blob path.

    Args:
        gsc_path: A full or partial path of GCS blob objects.
    Returns:
        bucket: The GCS bucket ID
        blob: Path to blob in GCS
    '''
    try:
        path_in_gcs = gcs_path.strip('gs://')
        bucket = path_in_gcs.split('/', 1)[0]
        blob = path_in_gcs.split('/', 1)[1]
        return bucket, blob
    except Exception as e:
        print(e)


def download_blob(bucket_name, blob_name, destination_file_name):
    '''
    Downloads a blob from the bucket.

    Args:
        bucket_name
        blob_name
        destination_file_name
    '''
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    assert blob.exists()
    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(
        blob_name,
        destination_file_name))


def upload_blob(bucket_name, blob_name, upload_fn):
    '''
    Uploads a file to a desired GCS location.

    Args:
        bucket_name
        blob_name
        upload_fn
    '''
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(upload_fn)

    print('Blob {} uploaded to {}.'.format(
        blob_name,
        upload_fn))

def process_features(features):
    '''
    Use to implement custom feature engineering logic, e.g. polynomial expansion, etc.

    Args:
        features - A pandas dataframe
    Returns:
        engineered_df
    '''

    # Implement min/max scaling
    scaler = MinMaxScaler()
    scaler.fit(features)
    engineered_features = pd.DataFrame(scaler.transform(features))
    return engineered_features