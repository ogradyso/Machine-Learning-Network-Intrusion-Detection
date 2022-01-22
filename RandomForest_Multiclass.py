import time 

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

#import category_encoders as ce
from sklearn.impute import SimpleImputer

from collections import Counter
# import the data:
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
# X_train['Label'] = y_train
# X_train['Label'] = X_train['Label'].astype('category').cat.codes

#X_train = X_res
#y_train = y_res

# # Remove features that are specific to this dataset:
X_train = X_train.drop('Timestamp', axis =1)
X_train = X_train.drop('Src IP', axis =1)
X_train = X_train.drop('Dst IP', axis =1)
X_train = X_train.drop('Flow ID', axis =1)
X_train = X_train.drop('Src Port', axis =1)
X_train = X_train.drop('Dst Port', axis =1)
X_train = X_train.drop('Protocol', axis =1)


X_train['Label'] = y_train
all_features = X_train.columns
numerical_features = X_train._get_numeric_data().columns

categorical_features = list(set(all_features) - set(numerical_features))

#high_activity_ports = X_train.groupby('Dst Port').count()
#high_activity_port_list = high_activity_ports[high_activity_ports['Label'] > 100].index

inifinity_cols = X_train[set(numerical_features)].columns.to_series()[np.isinf(X_train[set(numerical_features)]).any()]


#X_train['Label'] = ['Benign' if flow == 0 else 'Malicious' for flow in X_train['Label']]
#X_train['Label'] = y_res
X_train['Label'] = X_train['Label'].astype('category').cat.codes
y_train = X_train.pop('Label')
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
    
# Target encoding for categorical variables 
# X_train['Dst Port'] = X_train['Dst Port'].astype('category')
# X_train['Protocol'] = X_train['Protocol'].astype('category')

# tenc_port = ce.TargetEncoder()
# dst_port_targetEnc = tenc_port.fit_transform(X_train['Dst Port'],X_train['Flow Duration'])

# X_train = dst_port_targetEnc.join(X_train.drop('Dst Port', axis=1))

# tenc_protocol = ce.TargetEncoder()
# protocol_targetEnc = tenc_protocol.fit_transform(X_train['Protocol'],X_train['Flow Duration'])

# X_train = protocol_targetEnc.join(X_train.drop('Protocol', axis=1))

# # Inputer for numerical variables:
imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train = imputer_mean.fit_transform(X_train)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_train.shape

params = { 
    'n_estimators': [200, 500],
    'max_depth' : [4,6,8],
    'criterion' :['gini']
}

classifier = RandomForestClassifier()

gridsearch = GridSearchCV(classifier, params, cv=5, n_jobs=-1)

start_time = time.time()
rnd_frst_clf = gridsearch.fit(X_train, y_train)

end_time = time.time()

print("The optimization process took: {} hours".format((end_time - start_time)/3600))

rnd_frst_clf = gridsearch.fit(X_train, y_train)

