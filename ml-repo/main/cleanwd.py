# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 08:48:29 2018

@author: gbhat

Clean WatchDog Code w/ funcs
"""
# imports needed
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsOneClassifier
import numpy as np

#Set random seed so results stay consistent through runs
np.random.seed(42)
# Set data source
#csv_data = "https://raw.githubusercontent.com/V3SUV1US/ECG-WatchDog/master/ml-repo/main/datasets/arrythmia-ds.csv"
csv_data = "C:\\Users\\gbhat\\OneDrive\\Desktop\\Projects\\Coding\\Git\\ECG-WatchDog\\ml-repo\\main\\datasets\\arrythmia-ds.csv"
# Labels
heartdiseases = ('Normal', 'Ischemic changes', 'Old Anterior Myocardial Infarction', ' Old Inferior Myocardial Infarction ', 'Sinus tachycardy', 'Sinus bradycardy','Ventricular Premature Contraction',' Supraventricular Premature Contraction','Left bundle branch block','Right bundle branch block','Left ventricule hypertrophy','Atrial Fibrillation or Flutter', 'Others')
y_pos = np.arange(len(heartdiseases))
occurences = [245,44,15,15,13,25,3,2,9,50,4,5,22]

def load_data():
    """Returns data with na values labelled"""
    return pd.read_csv(csv_data, na_values='?')


raw_ecg_data = load_data()

def normalizeData(ecg_data_to_normalize):
    """Takes raw ECG data and returns a dataset without labels and
    with na values filled"""
    
    ecg_datawithoutlabel = ecg_data_to_normalize.drop('Identifier', axis=1)
    imputer = Imputer(missing_values="NaN", strategy='median')
    imputer.fit(ecg_datawithoutlabel)
    X = imputer.transform(ecg_datawithoutlabel)
    return X

def histOfData():
    plt.barh(y_pos, occurences, align='center', alpha=0.5)
    plt.yticks(y_pos, heartdiseases)
    plt.ylabel('Diseases in the Dataset')
    plt.xlabel('# of Occurences')
    plt.title('UCI Arrythmia Dataset Bar Graph')


X = normalizeData(raw_ecg_data)

def splitSets(X):
    shuffle_index = np.random.permutation(361)
    x_train, x_test, y_train, y_test = X[:361], X[361:], raw_ecg_data['Identifier'][:361], raw_ecg_data['Identifier'][361:]
    x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]
    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = splitSets(X)

ovo_clf= OneVsOneClassifier(SGDClassifier(max_iter=10, random_state=42, shuffle=True))
sgd_clf = SGDClassifier(max_iter=10, random_state=42, shuffle=True)
dt_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
randfor_clf = RandomForestClassifier(random_state=42, n_estimators=10)
svc_clf = svm.SVC(random_state=42, C=0.01)

x = [(ovo_clf, 'One Vs One SGD'), (sgd_clf, 'One Vs All SGD'), (dt_clf, 'Decision Tree'), (randfor_clf, 'Random Forest'), (svc_clf, 'Support Vector Classifier')]

def trainAndPlot(clf, name):
    clf.fit(x_train, y_train)
    train_score = cross_val_score(clf, x_train, y_train, cv=3, scoring='accuracy')
    test_score = cross_val_score(clf, x_test, y_test, cv=3, scoring='accuracy')
    mean_train_score = np.mean(train_score)
    mean_test_score = np.mean(test_score)

    print(name, 'train score')
    print(train_score)
    print(name, 'test score')
    print(test_score)


    #prediction_train = cross_val_predict(clf, x_train, y_train, cv=3)
    prediction_train = clf.predict(x_train)
    #prediction_test = cross_val_predict(clf, x_test, y_test, cv=3)
    prediction_test = clf.predict(x_test)

    train_confusion_matrix = confusion_matrix(y_train, prediction_train)
    train_row_sums = train_confusion_matrix.sum(axis=1, keepdims=True)
    norm_train_confusion_matrix = train_confusion_matrix/train_row_sums
    
    test_confusion_matrix = confusion_matrix(y_test, prediction_test)
    test_row_sums = test_confusion_matrix.sum(axis=1, keepdims=True)
    norm_test_confusion_matrix = test_confusion_matrix/test_row_sums


    
    #plt.xlabel("Actual Attribute")
    #plt.ylabel("Predicted Attirbute")
    
    plt.matshow(norm_train_confusion_matrix, cmap=plt.cm.cool)
    plt.colorbar()
    plt.matshow(norm_test_confusion_matrix, cmap=plt.cm.cool)

    index = np.arange(3)
    bar_width = 0.4
    
    fig, ax = plt.subplots()

    train_score_bars = plt.bar(index, train_score, bar_width, alpha = 0.3, color = 'b', label='Train Set Score')
    test_score_bars = plt.bar(index + bar_width, test_score, bar_width, alpha=0.3, color='r', label='Test Set Score')
    
    plt.xlabel('Fold Number')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy vs. Test Accuracy of ' + name)
    plt.xticks(index + (bar_width/2), ('F. 1', 'F. 2', 'F. 3'))
    plt.ylim([0,1])
    plt.legend()

    plt.show()

for i in x:
    trainAndPlot(i[0], i[1])

#trainAndPlot(randfor_clf, 'rf')
