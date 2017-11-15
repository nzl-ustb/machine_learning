from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
boston = load_boston()
print(boston.DESCR)
x = boston.data
y = boston.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)

print('max:', np.max(y))
print('min:', np.min(y))
print('average:', np.mean(y))
ss_x = StandardScaler()
ss_y = StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test = ss_y.transform(y_test.reshape(-1, 1))

lr = LinearRegression()
lr.fit(x_train, y_train)
lr_y_predict = lr.predict(x_test)
print('The Accuracy of LinearRegression is:', lr.score(x_test, y_test))
print('The value of R-squared of LinearRegression is :', r2_score(y_test, lr_y_predict))
print('The mean squared error of LinearRegression is :', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))
print('The mean absoluate error of LinearRegression is :', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))

sgdr = SGDRegressor()
sgdr.fit(x_train, y_train)
sgdr_y_predict = sgdr.predict(x_test)
print('The Accuracy of SGDRegressor is:', sgdr.score(x_test, y_test))
print('The value of R-squared of SGDRegressor is :', r2_score(y_test, sgdr_y_predict))
print('The mean squared error of SGDRegressor is :', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict)))
print('The mean absoluate error of SGDRegressor is :', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict)))


