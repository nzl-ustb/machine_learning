# 导入numpy, matplotlib, pandas, 用于数学运算，作图及数据分析
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# 从sklearn.cluster 中导入KMeans 模型
from sklearn.cluster import KMeans
# 从skleaen中导入度量函数库metrics
from sklearn import metrics
# 从sklearn.metrics中导入silhouette_score用于计算轮廓系数
from sklearn.metrics import silhouette_score

digits_train = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)
digits_test = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)

# 从训练与测试数据集上都分离出64维度的像素特征和1维度的数字目标
x_train = digits_train[np.arange(64)]
y_train = digits_train[64]

x_test = digits_test[np.arange(64)]
y_test = digits_test[64]

# 初始化,设置聚类中心数量为10
kmeans = KMeans(n_clusters=13)    # 这里要选取合适的类簇数
kmeans.fit(x_train)   # 注意这里跟之前的训练方式有些不同
# 逐条判断每个测试图像所属的聚类中心
y_pred = kmeans.predict(x_test)

# 使用ARI进行KMeans聚类性能评估
print(metrics.adjusted_rand_score(y_test, y_pred))







print(np.arange(5))

