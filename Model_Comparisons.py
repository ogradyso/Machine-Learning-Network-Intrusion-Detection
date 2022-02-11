#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 06:42:22 2022

This program loads the pickled models and compares their performance on the
finalized PCA test data.


@author: shauno
"""

import pickle
import pandas as pd

rnd_frst_mod = pickle.load(open('pca_rus_gs_rndFrst_model.pkl','rb'))


X_test_pca = pd.read_csv("X_test_pca.csv")
y_test_pca = pd.read_csv("y_test.csv")
X_test_pca.pop('Unnamed: 0')
y_test_pca.pop('Unnamed: 0')


