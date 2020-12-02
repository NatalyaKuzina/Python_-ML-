import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("mnist.csv")

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values
X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.20, train_size=0.8)
train, test_val = train_test_split(data, test_size=0.2)
X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5)
val, test = train_test_split(test_val, test_size=0.5)
print("Size of:")
print("- Training-set:\t\t{}".format(len(X_train)))
print("- Test-set:\t\t{}".format(len(X_val)))
print("- Validation-set:\t{}".format(len(X_test)))

u, counts=np.unique(y, return_counts=True)
u_train, counts_train=np.unique(y_train, return_counts=True)
u_val, counts_val=np.unique(y_val, return_counts=True)
u_test, counts_test=np.unique(y_test, return_counts=True)
print("labels train")
print(counts_train/counts)
print()
print("labels val")
print(counts_val/counts)
print()
print("labels test")
print(counts_test/counts)
