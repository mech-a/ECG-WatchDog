# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 13:00:39 2018

@author: gbhat
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(42)
csv_data = "https://raw.githubusercontent.com/V3SUV1US/ECG-WatchDog/master/ml-repo/main/datasets/arrythmia-ds.csv"

def load_data():
    return pd.read_csv(csv_data)
ecg_data = load_data()
#ecg_data.hist(bins=50, figsize=(20,15))
#plt.save_fig("attribute_histogram_plots")
ecg_data["Age"].hist()
plt.show()
train_set, test_set = train_test_split(ecg_data, test_size=0.2, random_state=42)