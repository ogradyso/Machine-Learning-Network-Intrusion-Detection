#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 06:42:22 2022

This program loads the pickled models and compares their performance on the
finalized PCA test data.


@author: shauno
"""

import joblib
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

X_test_pca = pd.read_csv('Data/X_train_pca_rusMajCmp_SlowHTTP.csv')
y_test_pca = pd.read_csv('Data/y_train_pca_rusMajCmp_SlowHTTP.csv')
X_test_pca.pop('Unnamed: 0')
y_test_pca.pop('Unnamed: 0')
X_test_pca['Label'] = y_test_pca
X_test_pca['Label'] = ['Web Attack' if flow == 'Brute Force -Web' or flow == 'SQL Injection' or flow == 'Brute Force -XSS' else flow for flow in X_test_pca['Label']]
#X_test_pca['Label'] = X_test_pca['Label'].astype('category').cat.codes
#y_test_pca = X_test_pca.pop('Label')


Sensitivity=pd.DataFrame()
Sensitivity['Label'] = X_test_pca['Label'].unique()

Specificity=pd.DataFrame()
Specificity['Label'] = X_test_pca['Label'].unique()

Precision=pd.DataFrame()
Precision['Label'] = X_test_pca['Label'].unique()

NegPredVal=pd.DataFrame()
NegPredVal['Label'] = X_test_pca['Label'].unique()

FalsePosRate=pd.DataFrame()
FalsePosRate['Label'] = X_test_pca['Label'].unique()

FalseNegRate=pd.DataFrame()
FalseNegRate['Label'] = X_test_pca['Label'].unique()

FalseDiscoveryRate=pd.DataFrame()
FalseDiscoveryRate['Label'] = X_test_pca['Label'].unique()

Accuracy=pd.DataFrame()
Accuracy['Label'] = X_test_pca['Label'].unique()

y_test_pca_text = X_test_pca.pop('Label')


# start for loop here:
# return all files as a list
for file in os.listdir(r'./PickledModels'):
    
    currentModel = joblib.load(r'./PickledModels/' + file)
    
    if (file.__contains__('xgboost')):
        X_test_pca['Label'] = y_test_pca_text
        X_test_pca['Label'] = ['Web Attack' if flow == 'Brute Force -Web' or flow == 'SQL Injection' or flow == 'Brute Force -XSS' else flow for flow in X_test_pca['Label']]
        X_test_pca['Label'] = X_test_pca['Label'].astype('category').cat.codes
        y_test_pca = X_test_pca.pop('Label')
    else:
        y_test_pca = y_test_pca_text
    
    # model predictions:
    y_predict = currentModel.predict(X_test_pca)

    cnf_matrix = confusion_matrix(y_test_pca, y_predict)
    print(cnf_matrix)

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

    filename = file.replace('.pkl','')
    Sensitivity[filename] = TPR
    Specificity[filename] = TNR
    Precision[filename] = PPV
    NegPredVal[filename] = NPV
    FalsePosRate[filename] = FPR
    FalseNegRate[filename] = FNR
    FalseDiscoveryRate[filename] = FDR
    Accuracy[filename] = ACC
    
pd.set_option('display.max_columns', 5)    
print(Sensitivity)

