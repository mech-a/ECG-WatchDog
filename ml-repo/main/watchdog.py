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
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve

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


curr_clf.fit(x_train, y_train_1)

y_train_predictions = cross_val_predict(curr_clf, x_train, y_train_1, cv=3)
<<<<<<< HEAD

confusionmat = confusion_matrix(y_train, y_train_predictions)
recall = recall_score(y_train_1, y_train_predictions)
precision = precision_score(y_train_1, y_train_predictions)
=======
y_scores = cross_val_predict(curr_clf, x_train, y_train_1, cv=3, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_1, y_scores)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])

plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-700000, 700000])
plt.savefig("precision_recall_vs_threshold_plot")
plt.show()
y_train_pred_90 = (y_scores > -500000)

#confusionmat = confusion_matrix(y_train, y_train_predictions)
recall = recall_score(y_train_1, y_train_pred_90)
precision = precision_score(y_train_1, y_train_pred_90)
>>>>>>> 41dc67742e6df11de58467e47234a63ba81b3916

# we need high precision low recall
