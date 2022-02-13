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

X_res['Label'] = y_res
X_res['Label'] = ['Web Attack' if flow == 'Brute Force -Web' or flow == 'SQL Injection' or flow == 'Brute Force -XSS' else flow for flow in X_res['Label']]
#X_test_pca['Label'] = X_test_pca['Label'].astype('category').cat.codes
y_res = X_res.pop('Label')


start_time = time.time()
baseline_rnd_frst_clf = baseLineClassifier.fit(X_res, y_res)

end_time = time.time()

# The baseline random forest training time is: 0.13481438563929662 hours
print("The baseline random forest training time is: {} hours".format((end_time - start_time)/3600))
joblib.dump(baseline_rnd_frst_clf, "pca_rus_bl_rndFrst_model.pkl")

start_time = time.time()
gridSearch_rnd_frst_clf = gridsearch.fit(X_res, y_res)

end_time = time.time()

# The grid search random forest training time is: 2.794910829199685 hours
print("The grid search random forest optimization time is: {} hours".format((end_time - start_time)/3600))


bestParameters = gridSearch_rnd_frst_clf.best_params_

print(bestParameters)
#{'criterion': 'gini', 'max_depth': 6, 'n_estimators': 200}

gridSearch_rnd_frst_clf = RandomForestClassifier(criterion='gini',max_depth=6,n_estimators=200)

start_time = time.time()
gridSearch_rnd_frst_clf = gridSearch_rnd_frst_clf.fit(X_res, y_res)

end_time = time.time()

# The grid search random forest training time is: 0.3208307798041238 hours
print("The grid search random forest training time is: {} hours".format((end_time - start_time)/3600))


joblib.dump(gridSearch_rnd_frst_clf, "pca_rus_gs_rndFrst_model.pkl")

