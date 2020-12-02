import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

data1 = pd.read_csv("hydrodynamics.csv")
data2 = pd.read_csv("noisysine.csv")

X = data1.iloc[:, :-1].values
Y = data1.iloc[:, -1].values
regr = DecisionTreeRegressor(random_state=0)
cross_val_score(regr, X, Y, cv=2)

x = data2.iloc[:, :-1].values
y = data2.iloc[:, -1].values
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(x, y)
regr_2.fit(x, y)

y_1 = regr_1.predict(x)
y_2 = regr_2.predict(x)
cross_val_score(regr_1, x, y, cv=2)

cross_val_score(regr_2, x, y, cv=2)

plt.scatter(x, y, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(x, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.plot(x, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
