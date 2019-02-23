from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# 准备数据
iris = load_iris()
#获取特征集和分类标识
feature = iris.data
labels = iris.target
# 随机抽取 33% 的数据作为测试集，其余为训练集
train_feature,test_feature,train_labels,test_labels = train_test_split(feature,labels,test_size = 0.33,random_state=0)
#创建 CART 分类树
clf = DecisionTreeClassifier(criterion='gini')
# 拟合构造 CART 分类树
clf = clf.fit(train_feature,train_labels)

# 用CART 分类树做预测
test_predict = clf.predict(test_feature)

# 预测结果和测试结果对比
score = accuracy_score(test_labels,test_predict)
print("CART 树的分类准确率 %.4lf" %score )