import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# 使用 scikit-learn.feature_extraction 中的特征转换器 DictVectorizer
from sklearn.feature_extraction import DictVectorizer
# 导入决策树分类器
from sklearn.tree import DecisionTreeClassifier
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
print(titanic.head())
# 查看数据的统计特性
print(titanic.info())

x = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']

#  对当前选择的特征进行探查
# print(x.info())

# 根据上面的输出，我们设计如下几个数据处理的任务：
# 1）age 这个数据列，只有633个，需要补完
# 2）sex 与 pclass 两个数据列的值都是类别型的，需要转化为数值特征，用0/1 代替

# 首先我们补充 age 里的数据，使用平均数或者中位数都是对模型偏离造成最小影响的策略
x['age'].fillna(x['age'].mean(), inplace=True)

# print(x.info())


# print(iris.DESCR)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)

# 特征抽取
vec = DictVectorizer(sparse=False)
# 特征转换后，发现凡是类别型的特征都单独剥离出来了，单独成一列特征，数值型的则保持不变
x_train = vec.fit_transform(x_train.to_dict(orient = 'record'))
print(vec.feature_names_)
x_test = vec.transform(x_test.to_dict(orient = 'record'))

# 使用单一决策树进行模型训练和预测
dtc = DecisionTreeClassifier()
dtc.fit(x_train,y_train)
dtc_y_pred = dtc.predict(x_test)

# 使用随机森林分类器
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfc_y_pred = rfc.predict(x_test)

# 使用梯度提升决策树
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(x_train, y_train)
gbc_y_pred = gbc.predict(x_test)

print('The accuracy of DecisionTree Classifier is:', dtc.score(x_test, y_test))
print(classification_report(y_test, dtc_y_pred, target_names=['died', 'survived']))

print('The accuracy of Random Forest Classifier is:', rfc.score(x_test, y_test))
print(classification_report(y_test, rfc_y_pred, target_names=['died', 'survived']))

print('The accuracy of Gradient tree boosting Classifier is:', gbc.score(x_test, y_test))
print(classification_report(y_test, gbc_y_pred, target_names=['died', 'survived']))
