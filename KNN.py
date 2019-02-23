"""
AUTHOR : SFZ
TIME : 2019-02-15
k-近邻分类算法
"""

from numpy import *
import operator

# 创建数据集
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels


# k-近邻分类算法
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount ={ }
    for i in range(k):
        votelabel = labels[sortedDistIndicies[i]]
        classCount[votelabel] = classCount.get(votelabel,0) + 1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]

#约会网站 : k-近邻分类算法

def file2matrix(filename):
    fr = open(filename)
    # 获取文本数据行数
    numberOfLines = len(fr.readlines())
    # 创建 n x 3 的零矩阵
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        # 截掉所有回车字符
        line = line.strip()
        listFromLine = line.split("\t")
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

# 归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)# 返回数组中每列最小的值
    maxVals = dataSet.max(0)
    range = maxVals-minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet-tile(minVals,(m,1))
    normDataSet = normDataSet/tile(range,(m,1))
    return normDataSet,range,minVals