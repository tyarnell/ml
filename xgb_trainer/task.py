import sys
import os
from datetime import datetime
import time

from argparse import ArgumentParser

import pandas as pd
from hypertune import HyperTune

from xgb_trainer.metadata import COLUMNS, FEATURES, TARGET, TRAIN_FILE, TRAIN_TEST_SPLIT, MODEL_FILE_NAME, N_SPLITS, SILENT, OBJECTIVE, BOOSTER, RANDOM_STATE, SHUFFLE, N_JOBS, SCORING, EARLY_STOPPING_ROUNDS
from xgb_trainer.input import split_gcs_path, download_blob, upload_blob, process_features
from xgb_trainer.model import fit_predict, fit_predict_cv, cv_fit

import pickle


def main(args):

    print('Downloading train & test data...')
    TRAIN_PATH = '{}/{}'.format(args.data_dir, TRAIN_FILE)
    BUCKET_NAME, TRAIN_BLOB = split_gcs_path(TRAIN_PATH)
    download_blob(BUCKET_NAME, TRAIN_BLOB, TRAIN_FILE)

    print()
    print('Loading data...')
    with open(os.path.join('.', TRAIN_FILE), 'r') as train_data:
        train_data = pd.read_hdf(os.path.join('.', TRAIN_FILE))

    X = train_data.drop([TARGET], axis=1)
    y = train_data[TARGET]
    
    X = process_features(X)

    HYPERPARAMS = {
        'silent': SILENT,
        'objective': OBJECTIVE,
        'booster': BOOSTER,
        'n_jobs': N_JOBS,
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'n_estimators': args.n_estimators,
        'gamma': args.gamma,
        'min_child_weight': args.min_child_weight,
        'max_delta_step': args.max_delta_step,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'colsample_bylevel': args.colsample_bylevel,
        'reg_alpha': args.reg_alpha,
        'reg_lambda': args.reg_lambda,
        'scale_pos_weight': args.scale_pos_weight
    }

    print()
    print('Training model...')
    if N_SPLITS == 1:
        model, results = fit_predict(HYPERPARAMS, X, y, TRAIN_TEST_SPLIT, RANDOM_STATE)
        score = results['accuracy_score']
    elif N_SPLITS > 1:
        model, _, score = fit_predict_cv(HYPERPARAMS, X, y, early_stopping_rounds=EARLY_STOPPING_ROUNDS, n_splits=N_SPLITS, shuffle=SHUFFLE, random_state=RANDOM_STATE)
        #model, mean_score = cv_fit(HYPERPARAMS, X, y, N_SPLITS, scoring=SCORING)
        #score = mean_score
    else:
        raise ValueError('N_SPLITS must be a positive integer. {} is not an acceptable value'.format(N_SPLITS))

    print()
    print('Reporting accuracy score...')
    hpt = HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='accuracy',
        metric_value=score,
        global_step=1000)

    print()
    print('Accuracy score: {}'.format(score))

    print()
    print('Saving and exporting model to GCS location...')
    model_fn = '{}_{}{}'.format(MODEL_FILE_NAME, time.time(), '.pkl')
    with open(model_fn, 'wb') as f:
        pickle.dump(model, f)

    DUMP_PATH = '{}/{}'.format(args.job_dir, model_fn)
    BUCKET_NAME, DUMP_BLOB = split_gcs_path(DUMP_PATH)
    upload_blob(BUCKET_NAME, DUMP_BLOB, model_fn)


if __name__ == "__main__":

    start = datetime.now()
    print()
    print('Training started at {}'.format(start))

    parser = ArgumentParser()
    parser.add_argument('--data-dir', help='GCS or local paths to training data', required=True)
    parser.add_argument('--job-dir', help='GCS location to write checkpoints and export models', required=True)
    parser.add_argument('--max-depth', default=3, type=int)
    parser.add_argument('--learning-rate', default=0.1, type=float)
    parser.add_argument('--n-estimators', default=100, type=int)
    parser.add_argument('--gamma', default=0, type=float)
    parser.add_argument('--min-child-weight', default=1, type=int)
    parser.add_argument('--max-delta-step', default=0, type=int)
    parser.add_argument('--subsample', default=1, type=float)
    parser.add_argument('--colsample-bytree', default=1, type=float)
    parser.add_argument('--colsample-bylevel', default=1, type=float)
    parser.add_argument('--reg-alpha', default=0, type=float)
    parser.add_argument('--reg-lambda', default=1, type=float)
    parser.add_argument('--scale-pos-weight', default=1, type=float)

    args = parser.parse_args()
    main(args)

    print()
    print('Training completed in {}'.format((datetime.now() - start)))
