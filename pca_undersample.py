#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 18:40:14 2022

@author: shaunogrady
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Standard Scaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
& matplotlib inline

# read in train  and test data

X_train = pd.read_csv('X_train')
y_train = pd.read_csv('y_train')

X_test = pd.read_csv('X_test')
y_test = pd.read_csv('y_test')







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


