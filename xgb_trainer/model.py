import sys
import os

import numpy as np
from scipy import interp

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report


def fit_predict(booster_params, X, y, test_size=0.2, random_state=None, print_score=False):
    '''Build an XGBoost classifier using the sklearn API.'''
    try: 
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size, random_state=random_state)
        classifier = XGBClassifier(**booster_params).fit(xtrain, ytrain)
        preds = classifier.predict(xtest)
        results = evaluate_model(ytest, preds, print_score=print_score)
        return classifier, results
    except Exception as e:
        raise e


def fit_predict_cv(booster_params, X, y, n_splits=3, shuffle=False, random_state=None, print_score=False):
    '''Build an XGBoost classifier using the sklearn API, along w/stratified Kfold CV.'''
    try:
        i = 0
        k_scores = {}
        mean_score = []
        classifier = XGBClassifier(**booster_params)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for train, test in cv.split(X, y):
            preds = classifier.fit(X[train], y[train]).predict(X[test])
            k_scores[i] = evaluate_model(y[test], preds, print_score=print_score)
            mean_score.append(accuracy_score(y[test], preds))
            i += 1
        classifier.fit(X, y)
        mean_score = sum(mean_score)/len(mean_score)
        return classifier, k_scores, mean_score
    except Exception as e:
        raise e


def evaluate_model(y_true, y_predict, print_score=False):
    '''Evaluate a models predictions, returning a dictionary of the necessary results.'''
    try:
        results = {}
        results['confusion_matrix'] = confusion_matrix(y_true, y_predict)
        results['classification_report'] = classification_report(y_true, y_predict)        
        results['accuracy_score'] = accuracy_score(y_true, y_predict) * 100
        if print_score:
            print()
            print("CONFUSION MATRIX:")
            print(results['Confusion Matrix'])
            print()
            print('CLASSIFICATION REPORT:')
            print(results['Classification Report'])
            print()        
            print("ACCURACY SCORE:")        
            print( "{} %".format(results['Accuracy Score']))
            print()
        return results
    except Exception as e:
        raise e