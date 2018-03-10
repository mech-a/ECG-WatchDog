# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 13:00:39 2018

@author: gbhat
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.linear_model import SGDClassifier

np.random.seed(42)
csv_data = "https://raw.githubusercontent.com/V3SUV1US/ECG-WatchDog/master/ml-repo/main/datasets/arrythmia-ds.csv"

def load_data():
    return pd.read_csv(csv_data, na_values= '?')
ecg_data = load_data()
#ecg_data.hist(bins=50, figsize=(20,15))
#plt.savefig("attribute_histogram_plots")
#ecg_data["Height"].hist()
#plt.show()
ecg_datawithoutlabel = ecg_data.drop('Identifier', axis=1)

imputer= Imputer(missing_values = "NaN", strategy= "mean")
X=imputer.fit_transform(ecg_datawithoutlabel)
#ecg_datawithoutlabel = ecg_data.drop('Age', axis = 0)

x_train, y_train = X, ecg_data['Identifier']
y_train_1= (y_train == 1)
sgd_clf= SGDClassifier(max_iter=5,random_state=42,shuffle=True)
Y = sgd_clf.fit(x_train,y_train_1)
