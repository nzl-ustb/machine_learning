from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
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
rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)
rfr_y_predict = rfr.predict(x_test)

etr = ExtraTreesRegressor()
etr.fit(x_train, y_train)
etr_y_predict = etr.predict(x_test)

gbr = GradientBoostingRegressor()
gbr.fit(x_train, y_train)
gbr_y_predict = gbr.predict(x_test)

print('R-squared value of RandomForestRegressor is :', rfr.score(x_test, y_test))
print('The mean squared error of RandomForestRegressor is :', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_predict)))
print('The mean absoluate error of RandomForestRegressor is :', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_predict)))
print(' ')
# etr这里出现问题，结果应该不对
print('R-squared value of ExtraTreesRegressor is :', etr.score(x_test, y_test))
print('The mean squared error of ExtraTreesRegressor is :', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_predict)))
print('The mean absoluate error of ExtraTreesRegressor is :', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_predict)))
print(' ')
# print(np.sort(etr.feature_importances_, boston.feature_names), axis = 0)
print(' ')
print('R-squared value of GradientBoostingRegressor is :', gbr.score(x_test, y_test))
print('The mean squared error of GradientBoostingRegressor is :', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_y_predict)))
print('The mean absoluate error of GradientBoostingRegressor is :', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_y_predict)))

