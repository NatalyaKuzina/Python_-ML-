import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data1 = pd.read_csv("hydrodynamics.csv")
data2 = pd.read_csv("noisysine.csv")

X = data1.iloc[:, :-1].values
Y = data1.iloc[:, -1].values
reg = LinearRegression().fit(X, Y)
y_pred = reg.predict(X)
reg.score(X, Y)
 
x = data2.iloc[:, :-1].values
y = data2.iloc[:, -1].values
reg2 = LinearRegression().fit(x, y)
y2_pred = reg2.predict(x)
reg2.score(x, y)
 
%matplotlib inline
plt.plot(x, y2_pred, 'r')
plt.scatter(x,y)
