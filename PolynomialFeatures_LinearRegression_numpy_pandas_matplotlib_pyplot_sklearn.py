import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

data1 = pd.read_csv("hydrodynamics.csv")
data2 = pd.read_csv("noisysine.csv")

X = data1.iloc[:, :-1].values
Y = data1.iloc[:, -1].values
polynomial_features= PolynomialFeatures(degree=2)
X_poly = polynomial_features.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, Y)
Y_poly_pred = model.predict(X_poly)
r1 = r2_score(Y,Y_poly_pred)
print(r1)
 
x = data2.iloc[:, :-1].values
y = data2.iloc[:, -1].values
polynomial_features2= PolynomialFeatures(degree=2)
x_poly = polynomial_features2.fit_transform(x)
model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)
r2 = r2_score(y,y_poly_pred)
print(r2)
 
plt.plot(x, y_poly_pred, 'r')
plt.scatter(x,y)

polynomial_features3= PolynomialFeatures(degree=3)
x3_poly = polynomial_features3.fit_transform(x)
model = LinearRegression()
model.fit(x3_poly, y)
y3_poly_pred = model.predict(x3_poly)
r3 = r2_score(y,y3_poly_pred)
print(r3)

plt.plot(x, y3_poly_pred, 'r')
plt.scatter(x,y)
