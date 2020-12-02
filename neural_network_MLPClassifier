import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

data = pd.read_csv("mnist.csv")

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values
X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.20, train_size=0.8)
X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5)
for n in ["logistic", "tanh", "relu"]:
    mlp = MLPClassifier(activation = n)
    mlp.fit(X_train, y_train)
    mlp.predict(X_train)
    print("Training set score: %f" % mlp.score(X_train, y_train))

for i in [1, 2, 3]:
    for j in [8, 16, 32, 64]:
        mlp = MLPClassifier(hidden_layer_sizes = (j)*i, activation = "logistic")
        mlp.fit(X_train, y_train)
        print("Kол-во внутренних слоев", i)
        print("Кол-во нейронов в слоях", j)
        print("Training set score: %f" % mlp.score(X_train, y_train))
        print("Val set score: %f" % mlp.score(X_val, y_val))
        print()

for i in [1, 2, 3]:
    for j in [8, 16, 32, 64]:
        mlp = MLPClassifier(hidden_layer_sizes = (j)*i, activation = "tanh")
        mlp.fit(X_train, y_train)
        print("Kол-во внутренних слоев", i)
        print("Кол-во нейронов в слоях", j)
        print("Training set score: %f" % mlp.score(X_train, y_train))
        print("Val set score: %f" % mlp.score(X_val, y_val))
        print()

for i in [1, 2, 3]:
    for j in [8, 16, 32, 64]:
        mlp = MLPClassifier(hidden_layer_sizes = (j)*i, activation =  "relu")
        mlp.fit(X_train, y_train)
        print("Kол-во внутренних слоев", i)
        print("Кол-во нейронов в слоях", j)
        print("Training set score: %f" % mlp.score(X_train, y_train))
        print("Val set score: %f" % mlp.score(X_val, y_val))
        print()
