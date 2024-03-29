import unittest

from xgb_trainer.model import fit_predict, fit_predict_cv

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

class TestModelMethods(unittest.TestCase):

    def test_fit_predict(self):
        # training inputs
        hyperparams = {}
        test_size = 0.2
        random_state = False

        # Load benchmark dataset, model and score
        X, y = load_breast_cancer(return_X_y=True)
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size, random_state=random_state)
        test_model = XGBClassifier(**hyperparams).fit(xtrain, ytrain)
        y_predict = test_model.predict(xtest)
        score = accuracy_score(ytest, y_predict)*100
        
        # Build test model and score
        model, results = fit_predict(hyperparams, X, y, test_size=test_size, random_state=random_state)

        d1_keys = vars(test_model).keys()
        d2_keys = vars(model).keys()

        self.assertEqual(d1_keys, d2_keys)
        self.assertEqual(score, results['accuracy_score'])

    def test_fit_predict_cv(self):
        # training inputs
        hyperparams = {}
        random_state = False
        n_splits = 10
        shuffle = False

        # Load benchmark dataset, model and score
        X, y = load_breast_cancer(return_X_y=True)
        
        test_score = []
        cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        test_model = XGBClassifier()
        for train_index, test_index in cv.split(X, y):
            eval_set = [(X[test_index], y[test_index])]
            preds = test_model.fit(X[train_index], y[train_index], eval_set=eval_set, early_stopping_rounds=100).predict(X[test_index])
            test_score.append(accuracy_score(y[test_index], preds))
        test_model = XGBClassifier(**hyperparams).fit(X, y)
        test_score = sum(test_score)/len(test_score)
        # Build test model and score
        modelcv, k_score, mean_score = fit_predict_cv(hyperparams, X, y, early_stopping_rounds=100,n_splits=n_splits, shuffle=shuffle)

        d1_keys = vars(test_model).keys()
        d2_keys = vars(modelcv).keys()

        print(d1_keys)
        print(d2_keys)

        exit()

        self.assertEqual(d1_keys, d2_keys)
        self.assertEqual(test_score, mean_score)

if __name__ == "__main__":
    unittest.main()
