import pandas as pd
import numpy as np
# 从sklearn.decomposition 中导入PCA
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report


digits_train = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)
digits_test = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)

# 从训练与测试数据集上都分离出64维度的像素特征和1维度的数字目标
x_train = digits_train[np.arange(64)]
y_train = digits_train[64]

x_test = digits_test[np.arange(64)]
y_test = digits_test[64]

# 初始化 对原始数据建模
svc = LinearSVC()
svc.fit(x_train, y_train)
y_predict = svc.predict(x_test)

# 使用PCA将64维度压缩到20个维度
estimator = PCA(n_components=20)


# 利用训练特征决定（fit）20个正交维度的方向，并转化（transfor）原训练特征
pca_x_train = estimator.fit_transform(x_train)
# 测试特征也按照上述20个正交维度方向进行转化
pca_x_test = estimator.transform(x_test)

pca_svc = LinearSVC()
pca_svc.fit(pca_x_train, y_train)
pca_y_predict = pca_svc.predict(pca_x_test)

# 原始数据评测
print(svc.score(x_test, y_test))
print(classification_report(y_test, y_predict, target_names=np.arange(10).astype(str)))

print(' ')
# PCA降维后评测
print(pca_svc.score(pca_x_test, y_test))
print(classification_report(y_test, pca_y_predict, target_names=np.arange(10).astype(str)))
