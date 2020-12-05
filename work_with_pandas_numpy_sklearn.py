import pandas as pd
import numpy as np

cancer = pd.read_csv('cancer.csv')
spam = pd.read_csv('spam.csv')

cancer.head()
spam.head()

from sklearn.model_selection import train_test_split
cancer_X, cancer_y = cancer.loc[:, '1':], cancer.loc[:,'label']
spam_X, spam_y = spam.loc[:, :'capital_run_length_total'], spam.loc[:,'label']
cancer_X
cancer_y

X_train, X_test, y_train, y_test = train_test_split(cancer_X, cancer_y, random_state=0)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train)

print(format(knn.score(X_test, y_test)))

X, y = np.array(cancer.loc[:, '1':]), np.array(cancer.loc[:,'label'])
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
loo = LeaveOneOut()
knn = knn.fit(X, y)
loo.get_n_splits(X)

k_neighbors = [1, 3, 5, 7, 9, 11]
for k in k_neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    scores = cross_val_score(knn, X, y, cv=loo)
    print("Ближайшие соседи", k)
    print("Итерации: ", len(scores))
    print("Средняя правильность: {:.2f}".format(scores.mean()))
    print()
    
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)

X_scaled = scaler.transform(X)
k_neighbors = [1, 3, 5, 7, 9, 11]
for k in k_neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_scaled, y)
    scores = cross_val_score(knn, X_scaled, y, cv=loo)
    print("Ближайшие соседи", k)
    print("Итерации: ", len(scores))
    print("Средняя правильность: {:.2f}".format(scores.mean()))
    print()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

print(format(knn.score(X_test, y_test)))

X, y = np.array(spam.loc[:, :'capital_run_length_total']), np.array(spam.loc[:,'label'])
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
loo = LeaveOneOut()
knn = knn.fit(X, y)
loo.get_n_splits(X)

k_neighbors = [1, 3, 5, 7, 9, 11]
for k in k_neighbors:
    knn= KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    scores = cross_val_score(knn, X, y, cv=loo)
    print("Ближайшие соседеи", k)
    print("Итераций: ", len(scores))
    print("Средняя правильность: {:.2f}".format(scores.mean()))
    print()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)

X_scaled = scaler.transform(X)
k_neighbors = [1, 3, 5, 7, 9, 11]
for k in k_neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_scaled, y)
    scores = cross_val_score(knn, X_scaled, y, cv=loo)
    print("Ближайшие соседеи", k)
    print("Итерации ", len(scores))
    print("Средняя правильность: {:.2f}".format(scores.mean()))
    print()
