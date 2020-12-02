import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
%matplotlib inline
data1 = pd.read_csv("spam_train.csv")
data2 = pd.read_csv("spam_test.csv")
X_train = data1.iloc[:, :-1].values
y_train = data1.iloc[:, -1].values
X_test = data2.iloc[:, :-1].values
y_test = data2.iloc[:, -1].values
import itertools
g = itertools.cycle(["r", "b", "g", "y"])
for n in range(2,11):
    clf = DecisionTreeClassifier(max_depth=n)
    clf.fit(X_train, y_train)
    clf.predict_proba(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, clf.predict_proba(X_test)[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.plot(fpr, tpr, color=next(g), label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.legend(loc="lower right")
    

for n in range(2,11):
    clf = DecisionTreeClassifier(max_depth=n)
    clf.fit(X_train, y_train)
    clf.predict_proba(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, clf.predict_proba(X_test)[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.plot(fpr, tpr, color= "g", label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.legend(loc="lower right")
