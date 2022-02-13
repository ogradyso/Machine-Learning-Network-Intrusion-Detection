# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 07:58:41 2022

@author: 12105
"""

import time
import joblib
import tempfile
import pickle

import numpy as np
import pandas as pd
import os
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import optuna

X_res = pd.read_csv('Data/X_train_pca_rusMajCmp_SlowHTTP.csv')
y_res = pd.read_csv('Data/y_train_pca_rusMajCmp_SlowHTTP.csv')
X_res.pop('Unnamed: 0')
y_res.pop('Unnamed: 0')

X_res['Label'] = y_res
X_res['Label'] = ['Web Attack' if flow == 'Brute Force -Web' or flow == 'SQL Injection' or flow == 'Brute Force -XSS' else flow for flow in X_res['Label']]
X_res['Label'] = X_res['Label'].astype('category').cat.codes
y_res = X_res.pop('Label')

baseline_xgboost_clf = XGBClassifier()

start_time = time.time()
baseline_xgboost_clf = baseline_xgboost_clf.fit(X_res, y_res)

end_time = time.time()

#The baseline xgboost training time is: 0.5992755153444078 hours
print("The baseline xgboost training time is: {} hours".format((end_time - start_time)/3600))
joblib.dump(baseline_xgboost_clf, "pca_rus_bl_xgboost_model.pkl")

try:
    study
except NameError:
    study = optuna.create_study(direction="maximize")

try:
    tempdir
except NameError:
    tempdir = tempfile.TemporaryDirectory().name
    os.mkdir(tempdir)

print(tempdir)

def objective(trial: optuna.trial.Trial) -> float:

    param = { # based on default suggestions from optuna website
        'objective': 'mlogloss',
        'booster': 'gbtree',
        'nthread': -1,
        'lambda': trial.suggest_loguniform('xgb_lambda', 1e-8, 1.0),
        'alpha': trial.suggest_loguniform('xgb_alpha', 1e-8, 1.0),
        'use_label_encoder': False,
        'tree_method':'hist'
    }

    clf = XGBClassifier(**param)
    
    with open(f"{os.path.join(tempdir, str(trial.number))}.pkl", "wb") as f:
        pickle.dump(clf, f)

    return np.mean(cross_val_score(clf, X_res, y_res, cv=8))

start_time = time.time()
study.optimize(objective, n_trials=10)

end_time = time.time()

#The optimization process took: 2.596759708589978 hours
print("The optimization process took: {} hours".format((end_time - start_time)/3600))

with open(f"{os.path.join(tempdir, str(study.best_trial.number))}.pkl", "rb") as f:
    best_model = pickle.load(f)
    
start_time = time.time()
best_model.fit(X_res,y_res)
end_time = time.time()

#The training process took: 0.03567716015709771 hours
print("The training process took: {} hours".format((end_time - start_time)/3600))

joblib.dump(best_model, "pca_rus_optuna_xgboost_model.pkl")

