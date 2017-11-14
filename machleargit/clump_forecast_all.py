import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.metrics import classification_report
#定义名字，为了，简单命名，第一列是序号，最后一列是输出
column_names=['sample code number','1','2','3','4','5','6','7','8','9','class']
#利用pandas从网上下载数据
data=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/'
                 'breast-cancer-wisconsin/breast-cancer-wisconsin.data',names=column_names)
#删除丢失的不完整的数据
data=data.replace(to_replace='?', value=np.nan)
data=data.dropna(how='any')
print(data.shape)
x_train, x_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]], test_size=0.25, random_state=33)
print(y_train.value_counts())
# 标准化数据，保证每个维度的特征数据方差为1，均值为0,。使得预测结果不会被某些维度过大的特征值而主导
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
# 初始化
lr = LogisticRegression()
sgdc = SGDClassifier()
# 调用LogisticRegression中的fit函数/模块用来训练模型参数
lr.fit(x_train,y_train)
# 使用训练好的模型lr对X_test进行预测，结果存储在lr_y_predict变量中
lr_y_predict = lr.predict(x_test)
# 调用SGDClassifier中的fit函数/模块用来训练模型参数
sgdc.fit(x_train, y_train)
# 使用训练好的模型sgdc对X_test进行预测，结果存储在sgdc_y_predict变量中
sgdc_y_predict = sgdc.predict(x_test)
print("Accuracy of LR Classifier:", lr.score(x_test, y_test))
print(classification_report(y_test, lr_y_predict, target_names=['Benign', 'Malignant']))
print("finish")
print("Accuracy of SGD Classifier:", sgdc.score(x_test, y_test))
print(classification_report(y_test, sgdc_y_predict, target_names=['Benign', 'Malignant']))
print("finish")
