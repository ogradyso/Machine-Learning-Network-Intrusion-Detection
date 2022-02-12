#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 06:42:22 2022

This program loads the pickled models and compares their performance on the
finalized PCA test data.


@author: shauno
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

rnd_frst_clf = joblib.load('pca_rus_gs_rndFrst_model.pkl')

X_test_pca = pd.read_csv('Data/X_train_pca_rusMajCmp_SlowHTTP.csv')
y_test_pca = pd.read_csv('Data/y_train_pca_rusMajCmp_SlowHTTP.csv')
X_test_pca.pop('Unnamed: 0')
#y_test_pca.pop('Unnamed: 0')
X_test_pca['Label'] = y_test_pca
X_test_pca['Label'] = ['Web Attack' if flow == 'Brute Force -Web' or flow == 'SQL Injection' or flow == 'Brute Force -XSS' else flow for flow in X_test_pca['Label']]
#X_test_pca['Label'] = X_test_pca['Label'].astype('category').cat.codes
y_test_pca = X_test_pca.pop('Label')

# model predictions:
y_predict = rnd_frst_clf.predict(X_test_pca)

cnf_matrix = confusion_matrix(y_test_pca, y_predict)
print(cnf_matrix)
#[[1 1 3]
# [3 2 2]
# [1 3 1]]

FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)