#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 22:11:51 2022

@author: shaunogrady
"""

import time
import pandas as pd

X_res = pd.read_csv('X_train_pca_rus.csv')
y_res = pd.read_csv('y_train_pca_rus.csv')
y_res.pop('Unnamed: 0')


start_time = time.time()

#model goes here

end_time = time.time()


print("The xgboost, optuna training time is: {} hours".format((end_time - start_time)/3600))

bestParameters = rnd_frst_clf.best_params_

print(bestParameters)


import joblib

joblib.dump(best_model, "pca_rus_gs_rndFrst_model.pkl")
