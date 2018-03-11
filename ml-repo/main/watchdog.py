# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 13:00:39 2018

@author: gbhat
"""

import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

np.random.seed(42)
csv_data = "https://raw.githubusercontent.com/V3SUV1US/ECG-WatchDog/master/ml-repo/main/datasets/arrythmia-ds.csv"

def load_data():
    return pd.read_csv(csv_data, na_values= '?')


ecg_data = load_data()

# ecg_data.hist(bins=50, figsize=(20,15))
# plt.savefig("attribute_histogram_plots")
# ecg_data["Height"].hist()
# plt.show()

ecg_datawithoutlabel = ecg_data.drop('Identifier', axis=1)

imputer= Imputer(missing_values="NaN", strategy="mean")
imputer.fit(ecg_datawithoutlabel)
# ecg_datawithoutlabel = ecg_data.drop('Age', axis = 0)
X = imputer.transform(ecg_datawithoutlabel)

shuffle_index= np.random.permutation(361)
x_train,x_test, y_train,y_test = X[:361],X[361:] ,ecg_data['Identifier'][:361],ecg_data['Identifier'][361:]
x_train,y_train= x_train[shuffle_index], y_train[shuffle_index]
y_train_1 = (y_train == 1)

sgd_clf = SGDClassifier(max_iter=5, random_state=42, shuffle=True)
dt_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
curr_clf = sgd_clf


curr_clf.fit(x_train, y_train)

y_train_predictions = cross_val_predict(curr_clf, x_train, y_train_1, cv=3)

confusionmat = confusion_matrix(y_train, y_train_predictions)
recall = recall_score(y_train, y_train_predictions)
precision = precision_score(y_train, y_train_predictions)

# we need high precision low recall
