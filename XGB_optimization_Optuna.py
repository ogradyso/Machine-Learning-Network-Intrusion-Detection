# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 17:33:43 2021

@author: 12105
"""
import time
#import logging
#import sys
import tempfile
import pickle
import os
#from functools import partial
#from warnings import simplefilter

import pandas as pd
import numpy as np
import optuna
import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from sklearn.exceptions import ConvergenceWarning
#from sklearn.model_selection import KFold
from xgboost import XGBClassifier

# import the data:
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
X_train['Label'] = y_train
X_train['Label'] = X_train['Label'].astype('category').cat.codes

#all_features = X_train.columns
numerical_features = X_train._get_numeric_data().columns

#categorical_features = list(set(all_features) - set(numerical_features))

high_activity_ports = X_train.groupby('Dst Port').count()
high_activity_port_list = high_activity_ports[high_activity_ports['Label'] > 100].index

inifinity_cols = X_train[set(numerical_features)].columns.to_series()[np.isinf(X_train[set(numerical_features)]).any()]

y_train = X_train.pop('Label')

# Replace the infinity with the maximum non-infinite value of a column multiplied by 2
# negative infinity will be replaced with a minimum value of a column multiplied by 2 
#(for negative numbers) or by the minimum non-negative value multiplied by negative 2:
for col_name in inifinity_cols:
    col_vals = X_train[col_name].replace(np.inf, 0)
    col_inf_replacement = col_vals.max() * 5
    col_vals = X_train[col_name].replace(-np.inf, 100000)
    if col_vals.min() > 0:
        col_negInf_replacement = col_vals.min() * -5
    elif  col_vals.min() == 0:
        col_negInf_replacement = -100000
    else:
        col_negInf_replacement = col_vals.min() * 5
    X_train.replace(np.inf, col_inf_replacement, inplace=True)
    X_train.replace(-np.inf, col_negInf_replacement, inplace=True)
    
# Target encoding for categorical variables 
X_train['Dst Port'] = X_train['Dst Port'].astype('category')
X_train['Protocol'] = X_train['Protocol'].astype('category')

tenc_port = ce.TargetEncoder()
dst_port_targetEnc = tenc_port.fit_transform(X_train['Dst Port'],X_train['Flow Duration'])

X_train = dst_port_targetEnc.join(X_train.drop('Dst Port', axis=1))

tenc_protocol = ce.TargetEncoder()
protocol_targetEnc = tenc_protocol.fit_transform(X_train['Protocol'],X_train['Flow Duration'])

X_train = protocol_targetEnc.join(X_train.drop('Protocol', axis=1))

# # Remove features that are specific to this dataset:
X_train = X_train.drop('Timestamp', axis =1)
X_train = X_train.drop('Src IP', axis =1)
X_train = X_train.drop('Dst IP', axis =1)
X_train = X_train.drop('Flow ID', axis =1)
X_train = X_train.drop('Src Port', axis =1)

# # Inputer for numerical variables:
imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train = imputer_mean.fit_transform(X_train)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train)

#simplefilter("ignore", category=ConvergenceWarning)
#simplefilter("ignore", category=RuntimeWarning)

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

# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# def objective(trial: optuna.trial.Trial) -> float:

#     param = { # based on default suggestions from optuna website
#         'objective': 'mlogloss',
#         'booster': trial.suggest_categorical('xgb_booster', ['gbtree', 'gblinear', 'dart']),
#         'lambda': trial.suggest_loguniform('xgb_lambda', 1e-8, 1.0),
#         'alpha': trial.suggest_loguniform('xgb_alpha', 1e-8, 1.0),
#         'use_label_encoder': False,
#         'tree_method':'hist'
#     }


#     clf = XGBClassifier(**param)
#     clf.fit(X_train, y_train)

#     with open(f"{os.path.join(tempdir, str(trial.number))}.pkl", "wb") as f:
#         pickle.dump(clf, f)

#     score = clf.score(X_val, y_val)

#     return score

# start_time = time.time()
# # attack_detection_model.fit(X_train_cv, y_train_cv, early_stopping_rounds=5,eval_set=[(X_test_cv, y_test_cv)])
# study.optimize(objective, n_trials=10)
# end_time = time.time()

#print("The optimization process took: {} hours".format((end_time - start_time)/3600))

with open(f"{os.path.join(tempdir, str(study.best_trial.number))}.pkl", "rb") as f:
    best_model = pickle.load(f)
    
# import the data:
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")
X_test['Label'] = y_test
X_test['Label'] = X_test['Label'].astype('category').cat.codes

#all_features = X_test.columns
numerical_features = X_test._get_numeric_data().columns

#categorical_features = list(set(all_features) - set(numerical_features))

high_activity_ports = X_test.groupby('Dst Port').count()
high_activity_port_list = high_activity_ports[high_activity_ports['Label'] > 100].index

inifinity_cols = X_test[set(numerical_features)].columns.to_series()[np.isinf(X_test[set(numerical_features)]).any()]

y_test = X_test.pop('Label')

# Replace the infinity with the maximum non-infinite value of a column multiplied by 2
# negative infinity will be replaced with a minimum value of a column multiplied by 2 
#(for negative numbers) or by the minimum non-negative value multiplied by negative 2:
for col_name in inifinity_cols:
    col_vals = X_test[col_name].replace(np.inf, 0)
    col_inf_replacement = col_vals.max() * 5
    col_vals = X_test[col_name].replace(-np.inf, 100000)
    if col_vals.min() > 0:
        col_negInf_replacement = col_vals.min() * -5
    elif  col_vals.min() == 0:
        col_negInf_replacement = -100000
    else:
        col_negInf_replacement = col_vals.min() * 5
    X_test.replace(np.inf, col_inf_replacement, inplace=True)
    X_test.replace(-np.inf, col_negInf_replacement, inplace=True)
    
# Target encoding for categorical variables 
X_test['Dst Port'] = X_test['Dst Port'].astype('category')
X_test['Protocol'] = X_test['Protocol'].astype('category')

tenc_port = ce.TargetEncoder()
dst_port_targetEnc = tenc_port.fit_transform(X_test['Dst Port'],X_test['Flow Duration'])

X_test = dst_port_targetEnc.join(X_test.drop('Dst Port', axis=1))

tenc_protocol = ce.TargetEncoder()
protocol_targetEnc = tenc_protocol.fit_transform(X_test['Protocol'],X_test['Flow Duration'])

X_test = protocol_targetEnc.join(X_test.drop('Protocol', axis=1))

# # Remove features that are specific to this dataset:
X_test = X_test.drop('Timestamp', axis =1)
X_test = X_test.drop('Src IP', axis =1)
X_test = X_test.drop('Dst IP', axis =1)
X_test = X_test.drop('Flow ID', axis =1)
X_test = X_test.drop('Src Port', axis =1)

# # Inputer for numerical variables:
imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
X_test = imputer_mean.fit_transform(X_test)

scaler = StandardScaler()

X_test = scaler.fit_transform(X_test)
X_test = pd.DataFrame(X_test)
    
test_predictions = best_model.predict(X_test)

accuracy_score(y_test, test_predictions)

