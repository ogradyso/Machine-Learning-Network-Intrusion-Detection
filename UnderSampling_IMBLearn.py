# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 14:52:12 2021

@author: 12105
"""
import numpy as np
import pandas as pd
#from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from collections import Counter
from sklearn.impute import SimpleImputer

# import the data:
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")

X_train.shape
#Retain timestamp

#'Timestamp',
X_train = X_train.drop(columns=['Flow ID','Src IP', 'Src Port','Dst IP','Dst Port', 'Protocol', 'Timestamp', 'Flow Duration'])


all_features = X_train.columns
numerical_features = X_train._get_numeric_data().columns

categorical_features = list(set(all_features) - set(numerical_features))

#high_activity_ports = X_train.groupby('Dst Port').count()
#high_activity_port_list = high_activity_ports[high_activity_ports['Label'] > 100].index

inifinity_cols = X_train[set(numerical_features)].columns.to_series()[np.isinf(X_train[set(numerical_features)]).any()]


#X_train['Label'] = y_res
#y_train = X_train.pop('Label_Bin')
#X_train.pop('Label_Bin')


#X_viz, test, y_viz, y_test = train_test_split(X_train, y_train, test_size=0.80, random_state=42)

# Replace the infinity with the maximum non-infinite value of a column multiplied by 2
# negative infinity will be replaced with a minimum value of a column multiplied by 2 
#(for negative numbers) or by the minimum non-negative value multiplied by negative 2:
for col_name in inifinity_cols:
    col_vals = X_train[col_name].replace(np.inf, 0)
    col_inf_replacement = col_vals.max() * 2
    col_vals = X_train[col_name].replace(-np.inf, 100000)
    if col_vals.min() > 0:
        col_negInf_replacement = col_vals.min() * -2
    elif  col_vals.min() == 0:
        col_negInf_replacement = -100000
    else:
        col_negInf_replacement = col_vals.min() * 2
    X_train.replace(np.inf, col_inf_replacement, inplace=True)
    X_train.replace(-np.inf, col_negInf_replacement, inplace=True)

#X_viz, test, y_viz, y_test = train_test_split(X_train, y_train, test_size=0.80, random_state=42)

imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train = imputer_mean.fit_transform(X_train)

X_train = pd.DataFrame(X_train)

# Undersample the majority class (Benign)
X_train['Label'] = y_train
X_train['ReLabel'] = ['Web Attack' if flow == 'Brute Force -Web' or flow == 'SQL Injection' or flow == 'Brute Force -XSS' else flow for flow in X_train['Label']]

X_train['Label'] = X_train['Label'].astype('category').cat.codes
X_train['Label_bin'] = ['Benign' if flow == 0 else 'Malicious' for flow in X_train['Label']]

y_train_bin = X_train.pop('Label_bin')


X_train['ReLabel'] = X_train['ReLabel'].astype('category').cat.codes

y_train = X_train['ReLabel']
print('Original dataset shape {}'.format(Counter(y_train)))

# rus = RandomUnderSampler(random_state=42)

# X_res, y_res = rus.fit_resample(X_train, y_train)
y_train_bin = X_train.pop('Label_bin')

nm = NearMiss()
X_res, y_res = nm.fit_resample(X_train, y_train_bin)


print('Resampled dataset shape {}'.format(Counter(y_res)))

y_train = X_train.pop('Label')

y_res_bin = y_res
y_res = y_train