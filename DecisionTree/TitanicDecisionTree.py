# 泰坦尼克号生存预测


# 模块1 ： 数据探索

import pandas as pd
# 数据加载
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 数据探索
print(train_data.info())
print('-'*30)
print(train_data.describe())
print('-'*30)
print(train_data.head())
print('-'*30)
print(train_data.tail())

print(" 探索测试数据 : ")

print(test_data.info())
print('-'*30)
print(test_data.describe())
print('-'*30)
print(test_data.head())
print('-'*30)
print(test_data.tail())

# 模块2 ： 数据清洗
#使用平均年龄来填充年龄中的nan值
train_data['Age'].fillna(train_data['Age'].mean(),inplace=True)
test_data['Age'].fillna(train_data['Age'].mean(),inplace=True)

# 使用票价的均值填充票价的nan值
train_data['Fare'].fillna(train_data['Fare'].mean(),inplace=True)
test_data['Fare'].fillna(train_data['Fare'].mean(),inplace=True)

print(train_data['Embarked'].value_counts())

# 使用登录最多的港口来填充港口的nan值
train_data['Embarked'].fillna('S',inplace=True)
test_data['Embarked'].fillna('S',inplace = True)


# 模块3 ：特征选择
feature = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
train_features = train_data[feature]
train_labels = train_data['Survived']
test_features = test_data[feature]

from sklearn.feature_extraction import DictVectorizer
devc = DictVectorizer(sparse=False)
train_features = devc.fit_transform(train_features.to_dict(orient='record'))
print(devc.feature_names_)

# 模块4 ：决策树模型
from sklearn.tree import DecisionTreeClassifier
# 构造ID3决策树
clf = DecisionTreeClassifier(criterion='entropy')
#决策树训练
clf.fit(train_features,train_labels)


# 模块5 ：模型预测 & 评估C:\Program Files (x86)\Graphviz2.38\bin
test_features = devc.transform(test_features.to_dict(orient='record'))
#决策树预测
pred_labels = clf.predict(test_features)

# 由于我们不知道真实的预测结果，所以无法用预测的值与真实的预测结果进行比较
#我们只能用训练集中的数据来进行模型评估
#得到决策树的准确率
acc_decision_tree = round(clf.score(train_features,train_labels),6)
print(u'score 准确率为 %.4lf' %acc_decision_tree)
#与自身来对数据集做准确率的统计，往往数据会不太准确，下面使用K-折交叉验证的方法来进行统计

import numpy as np
from sklearn.model_selection import cross_val_score
print(u'cross_val_score 准确率为 %.4lf' %np.mean(cross_val_score(clf,train_features,train_labels,cv=10)))

# 模块6 ： 决策树可视化
from sklearn import tree
import pydotplus
from sklearn.externals.six import StringIO
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("Titanic.pdf")