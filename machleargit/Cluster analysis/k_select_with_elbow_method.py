import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import pandas as pd

# 使用均匀分布函数随机三个类簇，每个簇周围10个数据样本
cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(5.5, 6.5, (2, 10))
cluster3 = np.random.uniform(3.0, 4.0, (2, 10))
cluster4 = np.random.uniform(2.0, 2.5, (2, 10))
# digits_train = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)
# x = digits_train[np.arange(64)]
# y_train = digits_train[64]
# 绘制30个数据样本的分布图像
x = np.hstack((cluster1, cluster2, cluster3, cluster4)).T
plt.scatter(x[:, 0], x[:, 1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# 测试9种不同聚类中心数量下，每种情况的聚类质量，并作图
K = range(1, 22)
meandistortions = []

for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(x)
    meandistortions.append(sum(np.min(cdist(x, kmeans.cluster_centers_, 'euclidean'), axis=1))/x.shape[0])

plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average Dispersion')
plt.title('Selecting k with the Elbow Method')
plt.show()
