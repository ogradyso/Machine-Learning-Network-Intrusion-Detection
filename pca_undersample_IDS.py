#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 18:40:14 2022

@author: shaunogrady
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.impute import SimpleImputer

#% matplotlib inline

# read in train  and test data

X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')

X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')


X_train = X_train.drop(columns=['Flow ID','Src IP', 'Src Port','Dst IP','Dst Port', 'Protocol', 'Timestamp', 'Flow Duration'])
X_test = X_test.drop(columns=['Flow ID','Src IP', 'Src Port','Dst IP','Dst Port', 'Protocol', 'Timestamp', 'Flow Duration'])


all_features = X_train.columns

numerical_features = X_train._get_numeric_data().columns

categorical_features = list(set(all_features) - set(numerical_features))

#high_activity_ports = X_train.groupby('Dst Port').count()
#high_activity_port_list = high_activity_ports[high_activity_ports['Label'] > 100].index

inifinity_cols = X_train[set(numerical_features)].columns.to_series()[np.isinf(X_train[set(numerical_features)]).any()]


X_categorical_train = X_train[categorical_features]
X_categorical_test = X_test[categorical_features]


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

#X_viz, test, y_viz, y_test = train_test_split(X_train, y_train, test_size=0.80, random_state=42)

imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train = imputer_mean.fit_transform(X_train)
X_test = imputer_mean.transform(X_test)


X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)


scaler = StandardScaler()

# standardize the features because PCA is sensitive to scale
scaler.fit(X_train)

# apply transform to both the training and set and the test set
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# apply PCA with a 0.95 variance
pca = PCA(0.95)

pca.fit(X_train)

pca.n_components_

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

X_train_pca = pd.DataFrame(X_train)
X_test_pca = pd.DataFrame(X_test)


# Reassemble PCA & Categorical  to save the data:


X_train_pca.to_csv("X_train_pca.csv")
y_train.to_csv("y_train_pca.csv")
X_test_pca.to_csv("X_test_pca.csv")
y_test.to_csv("y_test_pca.csv")

X_train_pca = pd.read_csv("X_train_pca.csv")
y_train_pca = pd.read_csv("y_train_pca.csv")
X_test_pca = pd.read_csv("X_test_pca.csv")
y_test_pca = pd.read_csv("y_test_pca.csv")
X_test_pca.pop('Unnamed: 0')
y_test_pca.pop('Unnamed: 0')
X_train_pca.pop('Unnamed: 0')
y_train_pca.pop('Unnamed: 0')


# Reassemble PCA & Categorical for IMB Learn?
X_train_pca['Label'] = y_train_pca
X_train_pca_targetClasses = X_train_pca[X_train_pca['Label'].isin(['FTP-BruteForce','Benign'])]
X_train_pca_allOther = X_train_pca[X_train_pca['Label'].isin(['DoS attacks-Hulk','DDOS attack-HOIC','DDoS attacks-LOIC-HTTP','DoS attacks-Slowloris','DoS attacks-SlowHTTPTest','Bot','SSH-Bruteforce','Infilteration','DoS attacks-GoldenEye','DDOS attack-LOIC-UDP','Brute Force -XSS','Brute Force -Web','SQL Injection'])]

X_train_pca_targetClasses['Label'] = X_train_pca_targetClasses['Label'].astype('category').cat.codes

X_train_pca_benign = ""

y_train = X_train_pca_targetClasses.pop('Label')

start_time = time.time()

#rus = RandomUnderSampler(sampling_strategy='majority',random_state=42)
#X_res, y_res = rus.fit_resample(X_train_pca_targetClasses, y_train)
ros = RandomOverSampler(sampling_strategy=0.1,random_state=42)
X_res, y_res = ros.fit_resample(X_train_pca_targetClasses, y_train)

end_time = time.time()


print("The undersample time is: {} hours".format((end_time - start_time)/3600))


X_res.shape

all_dfs = [X_res, X_train_pca_allOther]


X_train_pca_rusMajCmp_SlowHTTP = pd.concat(all_dfs).reset_index(drop=True)
y_train_pca_rusMajCmp_SlowHTTP = X_train_pca_rusMajCmp_SlowHTTP.pop('Label')

X_res['Label'] = y_res
X_res['Label'] = ['Benign' if recordLab == 0 else 'DoS attacks-SlowHTTPTest' for recordLab in X_res['Label']]
Counter(X_res['Label'])
#X_res.pop('Label')


X_res = pd.DataFrame(X_res)

X_res['Label'] = ['Benign' if recordLab == 0 else 'DoS attacks-SlowHTTPTest' for recordLab in X_res['Label']]

X_res.to_csv("X_train_pca_rus.csv")
y_res.to_csv("y_train_pca_rus.csv")
X_test_pca.to_csv("X_test_pca_rus.csv")
y_test.to_csv("y_test_pca_rus.csv")


np.random.seed(42)
rndperm = np.random.permutation(X_train.shape[0])


# Visualize the final data set
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=X_res.loc[rndperm,:],
    legend="full",
    alpha=0.3
)

#3 dimensional plot:
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=X_train.loc[rndperm,:]["pca-one"], 
    ys=X_train.loc[rndperm,:]["pca-two"], 
    zs=X_train.loc[rndperm,:]["pca-three"], 
    c=X_train.loc[rndperm,:]["y"], 
    cmap='tab10'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()

