# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 11:19:12 2021

@author: 12105
"""

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# import the data:
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")

X_train.shape
#Retain timestamp

#'Timestamp',
X_train = X_train.drop(columns=['Flow ID','Src IP', 'Src Port','Dst IP','Dst Port', 'Protocol', 'Flow Duration'])
X_viz = X_train
X_viz['Label'] = y_train
#X_train['Label'] = X_train['Label'].astype('category').cat.codes

#X_viz, test, y_viz, y_test = train_test_split(X_train, y_train, test_size=0.80, random_state=42)


X_viz['Label_Bin'] = ['1' if flow == 'Benign' else '0' for flow in X_viz['Label']]

#X_viz.to_csv('X_viz.csv',index=False)
#y_viz.to_csv('y_viz.csv',index=False)

malicious_train = X_viz[X_viz['Label'] != 'Benign']
malicious_train.shape

# Visualize/learn data:
sns.swarmplot(data = malicious_train, y = "Tot Fwd Pkts", x = "Label")

sns.swarmplot(data = malicious_train, y = "Tot Bwd Pkts", x = "Label")
sns.swarmplot(data = malicious_train, y = "Flow Duration", x = "Label")

sns.swarmplot(data = malicious_train, y = "Flow Duration", x = "Protocol")

sns.swarmplot(data = malicious_train, y = "Tot Fwd Pkts", x = "Label_Bin")

sns.swarmplot(data = malicious_train, y = "Tot Bwd Pkts", x = "Label_Bin")
sns.swarmplot(data = malicious_train, y = "Flow Duration", x = "Label_Bin")

sns.swarmplot(data = malicious_train, y = "Flow Duration", x = "Label_Bin")

#Time series Tot FWd Pkts by category
sns.lineplot(x="Timestamp", y = "Label", data = malicious_train)
#sns.lineplot(x = X_train["Timestamp"], y = "Pkt Len Mean", hue="Label", data = X_train)

# violin plot to look at Fwd Packets
Total_Fwd_Pckts_plt = sns.violinplot(x="Tot Fwd Pkts", y="Label", data=X_train)
Total_Bwd_Pckts_plt = sns.violinplot(x="Tot Bwd Pkts", y="Label", data=X_train)

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
    
violin_plot_multiple_limits('Label', X_train, sns.violinplot, [70000,10000,2000], 'Label', '')


### take a cross-section of network traffic for 2 hours before a labeled attack
#   and two hours after the attack

# use graphs above to get the following time:
time_of_attack ='20/02/2018 09:02:56'

from datetime import datetime
import pandas as pd
# Given timestamp in string
date_format_str = '%d/%m/%Y %H:%M:%S'
# create datetime object from timestamp string
given_time = datetime.strptime(time_of_attack, date_format_str)
print('Given Time: ', given_time)

n = 4
# Subtract 3 hours from datetime object
pre_attack_time = given_time - pd.DateOffset(hours=n)
print('Final Time (2 hours ahead of given time ): ', pre_attack_time)
# Convert datetime object to string in specific format 
pre_attack_time = pre_attack_time.strftime('%d/%m/%Y %H:%M:%S.%f')
print('Final Time as string object: ', pre_attack_time)

n = -4
# Subtract 3 hours from datetime object
post_attack_time = given_time - pd.DateOffset(hours=n)
print('Final Time (2 hours before given time ): ', post_attack_time)
# Convert datetime object to string in specific format 
post_attack_time = post_attack_time.strftime('%d/%m/%Y %H:%M:%S.%f')
print('Final Time as string object: ', post_attack_time)

X_train['Label']=y_train


print(Counter(y_train))
single_attack_train = X_train[((X_train['Timestamp'] >= pre_attack_time) & (X_train['Timestamp'] <= post_attack_time))]
timeseries_interval = single_attack_train[single_attack_train['Label']=="Malicious"]
single_attack_timeplot = sns.lineplot(x ="Timestamp", y = "Tot Fwd Pkts",hue='Label', data = single_attack_train)

###################################### Models:


y_train = X_train.pop('Label')
y_train = X_train.pop('Label_Bin')

all_features = X_train.columns

numerical_features = X_train._get_numeric_data().columns

categorical_features = list(set(all_features) - set(numerical_features))

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
    
from sklearn.impute import SimpleImputer

#y_train = X_train.pop('Label')
##y_train[y_train != 'Benign'] = "Malicious"

# # Remove features that are specific to this dataset:

# # Inputer for numerical variables:
imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train = imputer_mean.fit_transform(X_train)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train)

from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier

# define model
model = XGBClassifier(tree_method='hist')
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=1)
# evaluate model
scores = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.5f' % mean(scores))