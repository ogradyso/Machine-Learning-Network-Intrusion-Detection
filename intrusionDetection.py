# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 20:14:50 2021

@author: 12105
"""

import pandas as pd
import seaborn as sns

# import the data:
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")

X_train.describe(include='all')

#combine the features and labels for the following visualizations and data exploration steps
# This will be separated before we train the models
X_train['Label'] = y_train
X_train['Label'] = X_train['Label'].astype('category').cat.codes
X_train.head()

import matplotlib.pyplot as plt
#%matplotlib inline
plt.rcParams['figure.figsize'] = [20, 30]


def plot_multiple_limits(x_label, dataset, plot_type, limit_list, hue_label='', y_data=''):
    fig, axes = plt.subplots(nrows=len(limit_list), ncols=1)
    for i in range(len(limit_list)):
        if hue_label == '' and y_data == '':
            bar_plot = plot_type(ax=axes[i],x=x_label, data=dataset)
        elif y_data != '':
            bar_plot = plot_type(ax=axes[i],x=x_label, y=y_data, hue=hue_label, data=dataset)
            bar_plot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        else:
            bar_plot = plot_type(ax=axes[i],x=x_label, hue=hue_label, data=dataset)
            bar_plot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        bar_plot.set(ylabel='Record Count', xlabel='',ylim=(0, limit_list[i]))
        if i < len(limit_list)-1:
            bar_plot.set(xticklabels=[])
    for item in bar_plot.get_xticklabels():
            item.set_rotation(90)
    bar_plot.set(xlabel=x_label)
    
all_features = X_train.columns

numerical_features = X_train._get_numeric_data().columns

categorical_features = list(set(all_features) - set(numerical_features))
print("Categorical features: {}, {}, {}, {}, {}\n".format(*categorical_features))
print("Numerical features: {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},{}, {}, {}, {}, {},{}, {}, {}, {}, {},{}, {}, {}, {}, {},{}, {}, {}, {}, {},{}, {}, {}, {}, {}, {}, {}, {}, {}, {},{}, {}, {}, {}, {},{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(*set(numerical_features)))

plot_multiple_limits('Label', X_train, sns.countplot, [2000000, 500000, 2500])

from collections import Counter
y = y_train.squeeze()
counter = Counter(y.tolist())
print(counter)

high_activity_ports = X_train.groupby('Dst Port').count()
high_activity_port_list = high_activity_ports[high_activity_ports['Label'] > 100].index

plot_multiple_limits('Dst Port', X_train[X_train['Dst Port'].isin(high_activity_port_list)], sns.countplot, [2000000, 500000, 2500],'Label')

plot_multiple_limits('Protocol', X_train, sns.countplot, [2000000, 500000, 500], 'Label')


import numpy as np
inifinity_cols = X_train[set(numerical_features)].columns.to_series()[np.isinf(X_train[set(numerical_features)]).any()]

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
    
#Check to make sure the number of target labels is maintained:
len(X_train['Label'].unique())

import category_encoders as ce
# Target encoding for categorical variables 

X_train['Dst Port'] = X_train['Dst Port'].astype('category')
X_train['Protocol'] = X_train['Protocol'].astype('category')

tenc_port = ce.TargetEncoder()
dst_port_targetEnc = tenc_port.fit_transform(X_train['Dst Port'],X_train['Flow Duration'])

X_train = dst_port_targetEnc.join(X_train.drop('Dst Port', axis=1))

tenc_protocol = ce.TargetEncoder()
protocol_targetEnc = tenc_protocol.fit_transform(X_train['Protocol'],X_train['Flow Duration'])

X_train = protocol_targetEnc.join(X_train.drop('Protocol', axis=1))

from sklearn.impute import SimpleImputer

y_train = X_train.pop('Label')
##y_train[y_train != 'Benign'] = "Malicious"

# # Remove features that are specific to this dataset:
X_train = X_train.drop('Timestamp', axis =1)
X_train = X_train.drop('Src IP', axis =1)
X_train = X_train.drop('Dst IP', axis =1)
X_train = X_train.drop('Flow ID', axis =1)
X_train = X_train.drop('Src Port', axis =1)

# # Inputer for numerical variables:
imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train = imputer_mean.fit_transform(X_train)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train)

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import time

attack_detection_model = XGBClassifier()

X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X_train, y_train, random_state=42)

# print("Model training for default params took: {} hours".format((end_time - start_time)/3600))
# # Model training for default params took: 5.953236647248268 hours

attack_detection_model = XGBClassifier(tree_method = "hist")
start_time = time.time()
attack_detection_model.fit(X_train_cv, y_train_cv, early_stopping_rounds=5,eval_set=[(X_test_cv, y_test_cv)])
#optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
#optimize = partial(objective, X=X_train, y=y_train)
#study = optuna.create_study(direction='minimize')
#study.optimize(optimize, n_trials=100)
end_time = time.time()
      
print("Model training for tree_method='hist' took: {} hours".format((end_time - start_time)/3600))
#Model training for tree_method='hist' took: 0.2145266862048043 hours
      
#TODO: I need to turn this into a pipeline to reducce the code. I am just putting this in for now
# to get baseline model predictions for the binary classifier.


X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

#X_test.replace([np.inf, -np.inf],np.nan, inplace=True)

#X_test['Label'] = y_test

# Remove all rows with missing data:
#X_test = X_test.dropna()

#y_test = X_test.pop('Label')

#y_test[y_test != 'Benign'] = "Malicious"

# Replace the infinity with the maximum non-infinite value of a column multiplied by 2
# negative infinity will be replaced with a minimum value of a column multiplied by 2 
#(for negative numbers) or by the minimum non-negative value multiplied by negative 2:
for col_name in inifinity_cols:
    col_vals = X_test[col_name].replace(np.inf, 0)
    col_inf_replacement = col_vals.max() * 2
    col_vals = X_test[col_name].replace(-np.inf, 100000)
    if col_vals.min() > 0:
        col_negInf_replacement = col_vals.min() * -2
    elif  col_vals.min() == 0:
        col_negInf_replacement = -100000
    else:
        col_negInf_replacement = col_vals.min() * 2
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

# Remove features that are specific to this dataset:
X_test = X_test.drop('Timestamp', axis =1)
X_test = X_test.drop('Src IP', axis =1)
X_test = X_test.drop('Dst IP', axis =1)
X_test = X_test.drop('Flow ID', axis =1)
X_test = X_test.drop('Src Port', axis =1)

imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
X_test = imputer_mean.fit_transform(X_test)

scaler = StandardScaler()

X_test = scaler.fit_transform(X_test)
X_test = pd.DataFrame(X_test)

from sklearn.metrics import accuracy_score

predictions = attack_detection_model.predict(X_test)

accuracy_score(y_test, predictions)



#Test the following with GPU support (need to run without Anaconda):
#GPU hist method
# attack_detection_model_gpuhist = XGBClassifier(tree_method = "gpu_hist", gpu_id=0)
# start_time = time.time()
# attack_detection_model_gpuhist.fit(X_train_cv, y_train_cv, early_stopping_rounds=5,eval_set=[(X_test_cv, y_test_cv)])
# #optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
# #optimize = partial(objective, X=X_train, y=y_train)
# #study = optuna.create_study(direction='minimize')
# #study.optimize(optimize, n_trials=100)
# end_time = time.time()
      
# print("Model training for gpu_hist took: {} hours".format((end_time - start_time)/3600))

#GPU hist + single precision
# attack_detection_model_gpuhist_singPrec = XGBClassifier(tree_method = "gpu_hist", single_precision_histogram=True)
# start_time = time.time()
# attack_detection_model_gpuhist_singPrec.fit(X_train_cv, y_train_cv, early_stopping_rounds=5,eval_set=[(X_test_cv, y_test_cv)])
# #optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
# #optimize = partial(objective, X=X_train, y=y_train)
# #study = optuna.create_study(direction='minimize')
# #study.optimize(optimize, n_trials=100)
# end_time = time.time()
      
# print("Model training for gpu_hist with single precision took: {} hours".format((end_time - start_time)/3600))

      
# New Ideas for speeding up training:
#XGB Split by leaf (grow_policy = ‘lossguide’)
# LightGBM
# Distributed computing

