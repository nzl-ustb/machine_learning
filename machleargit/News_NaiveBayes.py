# 导入新闻数据抓取器
from sklearn.datasets import fetch_20newsgroups
# 从 sklearn.feature_extraction.text里导入用于文本特征向量转化模块 CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# 从 sklearn.naive_bayes里导入朴素贝叶斯模型 MultinomialNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# 即时从互联网下载数据
news = fetch_20newsgroups(subset='all')
print(len(news.data))
print(news.data[0])

x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)
vec = CountVectorizer()
x_train = vec.fit_transform(x_train)
x_test = vec.transform(x_test)

# 初始化贝叶斯模型
mnb = MultinomialNB()
# 利用训练数据对模型参数进行估计
mnb.fit(x_train,y_train)
y_predict = mnb.predict(x_test)

print('The Accuracy of Naive Bayes Classifier is:', mnb.score(x_test,y_test))
print(classification_report(y_test, y_predict, target_names=news.target_names))
