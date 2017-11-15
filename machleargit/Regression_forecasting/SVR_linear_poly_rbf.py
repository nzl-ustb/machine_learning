from sklearn.svm import SVR
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

# 使用线性核函数配置支持向量机
linear_svr = SVR(kernel='linear')
linear_svr.fit(x_train, y_train)
linear_svr_y_predict = linear_svr.predict(x_test)

# 使用多项式核函数配置支持向量机
poly_svr = SVR(kernel='poly')
poly_svr.fit(x_train, y_train)
poly_svr_y_predict = poly_svr.predict(x_test)

# 使用径向基核函数配置
rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(x_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(x_test)

print('R-squared value of linear SVR is :', linear_svr.score(x_test, y_test))
print('The mean squared error of linear SVR is :', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))
print('The mean absoluate error of linear SVR is :', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))
print(' ')
print('R-squared value of Poly SVR is :', poly_svr.score(x_test, y_test))
print('The mean squared error of Poly SVR is :', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))
print('The mean absoluate error of Poly SVR is :', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))
print(' ')
print('R-squared value of RBF SVR is :', rbf_svr.score(x_test, y_test))
print('The mean squared error of RBF SVR is :', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
print('The mean absoluate error of RBF SVR is :', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))

