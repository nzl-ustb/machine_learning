from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# 数据预处理
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

# 初始化
dtr = DecisionTreeRegressor()
dtr.fit(x_train, y_train)
dtr_y_predict = dtr.predict(x_test)

print('R-squared value of DecisionTreeRegressor is :', dtr.score(x_test, y_test))
print('The mean squared error of DecisionTreeRegressor is :', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_predict)))
print('The mean absoluate error of DecisionTreeRegressor is :', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_predict)))
print(' ')
# print('R-squared value of DecisionTreeRegressor is :', dtr.score(x_test, y_test))
# print('The mean squared error of DecisionTreeRegressor is :', mean_squared_error(y_test, dtr_y_predict))
# print('The mean absoluate error of DecisionTreeRegressor is :', mean_absolute_error(y_test, dtr_y_predict))
# print(' ')

