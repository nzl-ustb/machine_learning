import pandas as pd

df_train = pd.read_csv('D:/python/nzl/Machine Learning/Datasets/Breast-Cancer/breast-cancer-train.csv')
df_test = pd.read_csv('D:/python/nzl/Machine Learning/Datasets/Breast-Cancer/breast-cancer-test.csv')
df_test_negative = df_test.loc[df_test['Type'] == 0][['Clump Thickness', 'Cell Size']]
df_test_positive = df_test.loc[df_test['Type'] == 1][['Clump Thickness', 'Cell Size']]

import matplotlib.pyplot as plt

plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')

plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()

import numpy as np
intercept = np.random.random([1])
coef = np.random.random([2])
lx = np.arange(0, 12)
ly = (-intercept - lx * coef[0]) / coef[1]
# 绘制一条随机直线
plt.plot(lx, ly, c='yellow')
plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()

# 导入sklearn中的逻辑斯蒂回归分类器
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

# 使用前10条训练样本学习直线的系数和截距
lr.fit(df_train[['Clump Thickness', 'Cell Size']][:10], df_train['Type'][:10])
print('Testing accuracy(10 training samples):', lr.score(df_test[['Clump Thickness', 'Cell Size']],df_test['Type']))

intercept = lr.intercept_
coef = lr.coef_[0, :]
# 这个分类面应该是lx * coef[0] + ly * coef[1] +intercept = 0,映射到二维平面上之后，应该是：
ly = (-intercept - lx * coef[0])/coef[1]
plt.plot(lx, ly, c='green')
plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()

# 使用所有训练样本学习直线的系数和截距
lr.fit(df_train[['Clump Thickness', 'Cell Size']], df_train['Type'])
print('Testing accuracy(10 training samples):', lr.score(df_test[['Clump Thickness', 'Cell Size']],df_test['Type']))
intercept = lr.intercept_
coef = lr.coef_[0, :]
ly = (-intercept - lx * coef[0])/coef[1]
plt.plot(lx, ly, c='blue')
plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()

