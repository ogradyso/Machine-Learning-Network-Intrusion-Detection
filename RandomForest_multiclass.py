#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 12:15:11 2022

@author: shaunogrady
"""

import time
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators': [200, 500],
    'max_depth': [4,6,8],
    'criterion': ['gini']
}

baseLineClassifier = RandomForestClassifier()


classifier = RandomForestClassifier()

gridsearch = GridSearchCV(classifier, params, cv=5, n_jobs=-1, verbose=6)

X_res = pd.read_csv('Data/X_train_pca_rusMajCmp_SlowHTTP.csv')
y_res = pd.read_csv('Data/y_train_pca_rusMajCmp_SlowHTTP.csv')
X_res.pop('Unnamed: 0')
y_res.pop('Unnamed: 0')

start_time = time.time()
baseline_rnd_frst_clf = baseLineClassifier.fit(X_res, y_res)

end_time = time.time()

print("The baseline random forest training time is: {} hours".format((end_time - start_time)/3600))
joblib.dump(baseline_rnd_frst_clf, "pca_rus_bl_rndFrst_model.pkl")

start_time = time.time()
gridSearch_rnd_frst_clf = gridsearch.fit(X_res, y_res)

end_time = time.time()


print("The grid search random forest training time is: {} hours".format((end_time - start_time)/3600))

bestParameters = gridSearch_rnd_frst_clf.best_params_

print(bestParameters)


import joblib

joblib.dump(gridSearch_rnd_frst_clf, "pca_rus_gs_rndFrst_model.pkl")

