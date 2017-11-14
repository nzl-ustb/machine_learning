# 从 sklearn.datasets 里导入手写体数字加载器
from sklearn.datasets import load_digits

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# 从 sklearn.svm里导入基于线性假设的支持向量机分类器 LinearSVC
from sklearn.svm import LinearSVC
# 从通过数据加载器获得手写体数字的数码图像数据并存储在digits变量中
digits = load_digits()
print(digits.data.shape)
# 分割数据 75%用于训练 25%用于测试
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)
print(y_test.shape)
print(y_train.shape)

# 标准化
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

# 初始化
lsvc = LinearSVC()
# 进行模型训练
lsvc.fit(x_train, y_train)
y_predict = lsvc.predict(x_test)

# 评估
print('The Accuracy of Linear SVC is:', lsvc.score(x_test,y_test))

print(classification_report(y_test, y_predict, target_names=digits.target_names.astype(str)))

