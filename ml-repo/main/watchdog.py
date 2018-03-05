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
#plt.save_fig("attribute_histogram_plots")
ecg_data["Height"].hist()
plt.show()
imputer= Imputer(strategy= "median")
imputer.fit(ecg_data)
ecg_datawithoutlabel = ecg_data.drop('Identifier', axis=1)
ecg_datawithoutlabel.as_matrix()
#x_train, y_train = ecg_datawithoutlabel, ecg_data['Identifier']
#sgd_clf= SGDClassifier(random_state=42)
#sgd_clf.fit(x_train,y_train)
