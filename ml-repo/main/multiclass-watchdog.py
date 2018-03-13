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

np.random.seed(42)
csv_data = "https://raw.githubusercontent.com/V3SUV1US/ECG-WatchDog/master/ml-repo/main/datasets/arrythmia-ds.csv"


def load_data():
    return pd.read_csv(csv_data, na_values='?')


ecg_data = load_data()

# ecg_data.hist(bins=50, figsize=(20,15))
# plt.savefig("attribute_histogram_plots")
# ecg_data["Height"].hist()
# plt.show()


def normalizeData(ecg_data_to_clean):
    ecg_datawithoutlabel = ecg_data_to_clean.drop('Identifier', axis=1)
    imputer = Imputer(missing_values="NaN", strategy='median')
    imputer.fit(ecg_datawithoutlabel)
    # ecg_datawithoutlabel = ecg_data.drop('Age', axis = 0)
    X = imputer.transform(ecg_datawithoutlabel)
    return X
    

X = normalizeData(ecg_data)
heartdiseases = ('Normal', 'Ischemic changes', 'Old Anterior Myocardial Infarction', ' Old Inferior Myocardial Infarction ', 'Sinus tachycardy', 'Sinus bradycardy','Ventricular Premature Contraction',' Supraventricular Premature Contraction','Left bundle branch block','Right bundle branch block','Left ventricule hypertrophy','Atrial Fibrillation or Flutter', 'Others')
y_pos = np.arange(len(heartdiseases))
occurences = [245,44,15,15,13,25,3,2,9,50,4,5,22]
 
plt.barh(y_pos,occurences, align='center', alpha=0.5)
plt.yticks(y_pos, heartdiseases)
plt.ylabel('Diseases in the Dataset')
plt.title('UCI Arrythmia Dataset Bar Graph')
plt.show()

shuffle_index = np.random.permutation(361)
x_train, x_test, y_train, y_test = X[:361], X[361:], ecg_data['Identifier'][:361], ecg_data['Identifier'][361:]
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

# when we want to only check for the label 1, use this
y_train_1 = (y_train == 1)

# these are the different classifiers we want to use
ovo_clf= OneVsOneClassifier(SGDClassifier(max_iter=400, random_state=42, shuffle=True))
sgd_clf = SGDClassifier(max_iter=400, random_state=42, shuffle=True)
dt_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
randfor_clf = RandomForestClassifier(random_state=42, n_estimators=10)
svc_clf = svm.SVC(random_state=42)
curr_clf = randfor_clf

# ALL OF THE FOLLOWING ARE USING NORMAL PREDICT NOT CROSS VALUE PREDICT
# svm svc is really good! 100% accuracy on x train.
# sgd clf gets better over iterations. figure out how to map that. 400 was decent
# dt clf w these params is horrible. like rly bad.
# randfor is good. the gray on 7th row 1st col is because there's only 3 instances in train datasets; got 2correct/3


randfor_clf.fit(x_train, y_train)
ovo_clf.fit(x_train, y_train)
sgd_clf.fit(x_train, y_train)
dt_clf.fit(x_train, y_train)
svc_clf.fit(x_train, y_train)

#y_train_predictions = cross_val_predict(curr_clf, x_train, y_train, cv=3)
accuracyofOvOSGDbartrain = cross_val_score(ovo_clf, x_train, y_train, cv=3, scoring= "accuracy")
accuracyofrfbartrain = cross_val_score(randfor_clf, x_train, y_train, cv=3, scoring= "accuracy")
accuracyofsgdbartrain = cross_val_score(sgd_clf, x_train, y_train, cv=3, scoring= "accuracy")
accuracyofdtbartrain = cross_val_score(dt_clf, x_train, y_train, cv=3, scoring= "accuracy")
accuracyofsvcbartrain = cross_val_score(svc_clf, x_train, y_train, cv=3, scoring= "accuracy")
accuracyofovobartest = cross_val_score(ovo_clf, x_test, y_test, cv=3, scoring= "accuracy")
accuracyofrfbartest = cross_val_score(randfor_clf, x_test, y_test, cv=3, scoring= "accuracy")
accuracyofsgdbartest = cross_val_score(sgd_clf, x_test, y_test, cv=3, scoring= "accuracy")
accuracyofdtbartest = cross_val_score(dt_clf, x_test, y_test, cv=3, scoring= "accuracy")
accuracyofsvcbartest = cross_val_score(svc_clf, x_test, y_test, cv=3, scoring= "accuracy")
print('OvO train' )
print(accuracyofOvOSGDbartrain)
print('rf train')
print(accuracyofrfbartrain)
print('sgd train')
print(accuracyofsgdbartrain)
print('dt train')
print(accuracyofdtbartrain)
print('svc train')
print(accuracyofsvcbartrain)
print('OvO test')
print(accuracyofovobartest)
print('rf test')
print(accuracyofrfbartest)
print('sgd test')
print(accuracyofsgdbartest)
print('dt test')
print(accuracyofdtbartest)
print('svc test')
print(accuracyofsvcbartest)
#OvO,rf, svc
accuracyofOvOSGDbartrainmean= np.mean(accuracyofOvOSGDbartrain)
accuracyofrfbartrainmean= np.mean(accuracyofrfbartrain)
accuracyofsvcbartrainmean= np.mean(accuracyofsvcbartrain)

accuracyofovobartestmean= np.mean(accuracyofovobartest)
accuracyofrfbartestmean = np.mean(accuracyofrfbartest)
accuracyofsvcbartestmean = np.mean(accuracyofsvcbartest)









accuracyofOvOSGD = cross_val_predict(ovo_clf, x_train, y_train, cv=3)
accuracyofSGD = cross_val_predict(sgd_clf, x_train, y_train, cv=3)
accuracyofdt = cross_val_predict(dt_clf, x_train, y_train, cv=3)
accuracyofrf = cross_val_predict(randfor_clf, x_train, y_train, cv=3)
accuracyofsvc = cross_val_predict(svc_clf, x_train, y_train, cv=3)
testpredsrf = cross_val_predict(randfor_clf, x_test, y_test, cv=3)
test_predsovo = cross_val_predict(ovo_clf, x_test, y_test, cv=3)
test_predssgd = cross_val_predict(sgd_clf, x_test, y_test, cv=3)
test_predsdt = cross_val_predict(dt_clf, x_test, y_test, cv=3)
test_predssvc = cross_val_predict(svc_clf, x_test, y_test, cv=3)



confusionmatrftrain = confusion_matrix(y_train, accuracyofrf)
confusionmatovotrain = confusion_matrix(y_train, accuracyofOvOSGD)
confusionmatSGDtrain = confusion_matrix(y_train, accuracyofSGD)
confusionmatdttrain = confusion_matrix(y_train, accuracyofdt)
confusionmatsvctrain = confusion_matrix(y_train, accuracyofsvc)

confusionmatrftest = confusion_matrix(y_test, testpredsrf)
confusionmatovotest = confusion_matrix(y_test, test_predsovo)
confusionmatsgdtest = confusion_matrix(y_test, test_predssgd)
confusionmatdttest = confusion_matrix(y_test, test_predsdt)
confusionmatsvctest = confusion_matrix(y_test, test_predssvc)







#row_sums = confusionmat.sum(axis=1, keepdims=True)
#norm_confusion_matrix = confusionmat/row_sums

plt.matshow(confusionmatrftrain, cmap=plt.cm.pink)
plt.matshow(confusionmatovotrain, cmap=plt.cm.pink)
plt.matshow(confusionmatSGDtrain, cmap=plt.cm.pink)
plt.matshow(confusionmatdttrain, cmap=plt.cm.pink)
plt.matshow(confusionmatsvctrain, cmap=plt.cm.pink)
plt.matshow(confusionmatrftest, cmap=plt.cm.pink)
plt.matshow(confusionmatovotest, cmap=plt.cm.pink)
plt.matshow(confusionmatsgdtest, cmap=plt.cm.pink)
plt.matshow(confusionmatdttest, cmap=plt.cm.pink)
plt.matshow(confusionmatsvctest, cmap=plt.cm.pink)
plt.show()




# we need high recall low precision for binary classifier
#Possibly implement GridSearchCV or RandomizedSearchCV to change hyperparameters
# following is using a scaled train set
# result: not useful for random forest classifier, maybe useful (untested) for sgd.
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.astype(np.float64))
accuracyofOvOSGDscaled = cross_val_score(ovo_clf, x_train_scaled, y_train, cv=3, scoring="accuracy")
accuracyofSGDscaled = cross_val_score(sgd_clf, x_train_scaled, y_train, cv=3, scoring="accuracy")
accuracyofdtscaled = cross_val_score(dt_clf, x_train_scaled, y_train, cv=3, scoring="accuracy")
accuracyofrfscaled = cross_val_score(randfor_clf, x_train_scaled, y_train, cv=3, scoring="accuracy")
accuracyofsvcscaled = cross_val_score(svc_clf, x_train_scaled, y_train, cv=3, scoring="accuracy")



x_test_scaled = scaler.fit_transform(x_test.astype(np.float64))
testscalerscoreofOvO = cross_val_score(ovo_clf, x_test_scaled, y_test, cv=3, scoring='accuracy')
testscalerscoreofSGD = cross_val_score(sgd_clf, x_test_scaled, y_test, cv=3, scoring='accuracy')
testscalerscoreofdt = cross_val_score(dt_clf, x_test_scaled, y_test, cv=3, scoring='accuracy')
testscalerscoreofrf = cross_val_score(randfor_clf, x_test_scaled, y_test, cv=3, scoring='accuracy')
testscalerscoreofsvc = cross_val_score(svc_clf, x_test_scaled, y_test, cv=3, scoring='accuracy')
print('Scaled')
print('OvO train' )
print(accuracyofOvOSGDscaled)
print('SGD train')
print(accuracyofSGDscaled)
print('dt train')
print(accuracyofdtscaled)
print('rf train')
print(accuracyofrfscaled)
print('svc train')
print(accuracyofsvcscaled)
print('OvO test')
print(testscalerscoreofOvO)
print('SGD test')
print(testscalerscoreofSGD)
print('dt test')
print(testscalerscoreofdt)
print('rf test')
print(testscalerscoreofrf)
print('svc test')
print(testscalerscoreofsvc)

trainaccuracyofOvOSGDscaledmean= np.mean(accuracyofOvOSGDscaled)
trainaccuracyofrfscaledmean= np.mean(accuracyofrfscaled)
trainaccuracyofsvcscaledmean= np.mean(accuracyofsvcscaled)

testscalerscoreofOvOmean = np.mean(testscalerscoreofOvO)
testscalerscoreofrfmean = np.mean(testscalerscoreofrf)
testscalerscoreofsvcmean = np.mean(testscalerscoreofsvc)

trainunscaled = (accuracyofOvOSGDbartrainmean,accuracyofrfbartrainmean,accuracyofsvcbartrainmean)
trainscaled=(trainaccuracyofOvOSGDscaledmean,trainaccuracyofrfscaledmean,trainaccuracyofsvcscaledmean)

testunscaled=(accuracyofovobartestmean,accuracyofrfbartestmean,accuracyofsvcbartestmean)
testscaled=(testscalerscoreofOvOmean,testscalerscoreofrfmean,testscalerscoreofsvcmean)

fig, ax = plt.subplots()
index = np.arange(3)
bar_width = 0.4
 
rects1 = plt.bar(index, trainunscaled, bar_width,alpha=0.5,
                 color='r',
                 label='Unscaled')
 
rects2 = plt.bar(index + bar_width, trainscaled, bar_width,alpha=0.1,
                 color='r',
                 label='Scaled')
 
plt.xlabel('Types of Classifiers')
plt.ylabel('Mean Accuracy')
plt.title('Scaled vs. Unscaled for train set')
plt.xticks(index + 0.2, ('OvO', 'Rf', 'SVC' ))
plt.legend()
 

plt.show()
fig2, ax2 = plt.subplots()
index = np.arange(3)
bar_width = 0.4
 
rects3 = plt.bar(index, testunscaled, bar_width,alpha=0.5,
                 color='r',
                 label='Unscaled')
 
rects4 = plt.bar(index + bar_width, testscaled, bar_width,alpha=0.1,
                 color='r',
                 label='Scaled')
 
plt.xlabel('Types of Classifiers')
plt.ylabel('Mean Accuracy')
plt.title('Scaled vs. Unscaled for test set')
plt.xticks(index + 0.2, ('OvO', 'Rf', 'SVC' ))
plt.legend()
 

plt.show()









