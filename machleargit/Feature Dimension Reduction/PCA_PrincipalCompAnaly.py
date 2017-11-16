import pandas as pd
import numpy as np
# 从sklearn.decomposition 中导入PCA
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


digits_train = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)
digits_test = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)

# 分割训练数据的特征向量和标记
x_digits = digits_train[np.arange(64)]
y_digits = digits_train[64]

# 初始化一个可以将高维度特征向量（六十四维）压缩至二个维度的 PCA
estimator = PCA(n_components=2)
x_pca = estimator.fit_transform(x_digits)

# 显示10类手写体数字图片经PCA压缩后的二维空间分布
def plot_pca_scatter():
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']

    for i in range(len(colors)):
        px = x_pca[:, 0][y_digits.as_matrix() == i]  # 这里的[y_digits.as_matrix]主要对x_pca第一列的所有行起到选择作用，也就是说假设i=0时，
        py = x_pca[:, 1][y_digits.as_matrix() == i]  # 选择出所有训练样本的标签为0的x_pca，并将其用二维图展现出来，不同的数字用不同的颜色画出来
        plt.scatter(px, py, c=colors[i])             # 最后，通过最终效果图可以发现，同一类型的digits基本上分布在同一块区域

    plt.legend(np.arange(0, 10).astype(str))  # 图例的设置
    plt.xlabel('First Principal Component')
    plt.ylabel('Sencond Principal Component')
    plt.show()

plot_pca_scatter()

