# -*- coding: UTF-8 -*-
from numpy import * #载入numpy库
import numpy as np
import operator # 载入operator模块
import sys

#简单kNN分类器
def classify0(inX,dataSet,labels,k):
    # 首先用欧式距离法计算已知类别数据集与当前点的距离
    dataSetSize=dataSet.shape[0] # 读取数据集的行数，并把行数放到dataSetSize里，shape[]用来读取矩阵的行列数，shape[1]表示读取列数
    diffMat=tile(inX,(dataSetSize,1))-dataSet # tile(inX,(dataSetSize,1))复制比较向量inX，tile的功能是告诉inX需要复制多少遍，这里复制成(dataSetSize行，1次)
                                              # 目的是把inX转化成与数据集相同大小，再与数据集矩阵相减，形成的差值矩阵存放在diffMat里
    sqDiffMat=diffMat**2 # 把矩阵里的各个元素依次平方
    sqDistances=sqDiffMat.sum(axis=1) # 实现计算结果，axis=1表矩阵每一行元素相加
    distances=sqDistances**0.5 # 开根号

    # 选择距离最小的k个点
    sortedDisIndicies=distances.argsort() # 使用argsort排序，返回从小到大到“顺序值”,如{2,4,1}返回{1,2,0}，依次为其顺序到索引
    classCount={} # 新建一个字典，用于计数
    for i in range(k): # 按顺序对标签进行计数
        voteIlabel=labels[sortedDisIndicies[i]] # 第i名对应的类别
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1 # 计数:归类为A（B）的次数，并给字典赋值

    # 按归类次数排序
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True) # classCount.items()把字典转化为列表，每个元素是一个tuple，tuple的第一个元素是键，第二个元素是值
                                                                                        # key=operator.itemgetter(1)按位置1元素降序排列
    return sortedClassCount[0][0] # 返回出现次数最多到label值，即为当前点的预测分类

#1.将文本转换为矩阵
def file2matrix(filename):
    with open(filename) as fp:
        Arr_lines = fp.readlines() #读取全部行
        number = len(Arr_lines) #行数
        #初始化数据为m行3列（飞行里程，游戏时间，冰淇淋数）
        #标签单独创建一个向量保存
        return_mat = zeros((number, 3))
        label_vec = []
        index = 0

        for line in Arr_lines:
            line = line.strip()
            listFromLine = line.split('\t')  #按换行符分割数据
            #将文本数据前三行存入数据矩阵，第四行存入标签向量
            return_mat[index,:] = listFromLine[0:3]
            label_vec.append(int(listFromLine[3]))
            index += 1
    return return_mat, label_vec
#输出
#return_mat, label_vec=file2matrix('datingTestSet2.txt')
#print(return_mat, label_vec)

'''
#2.画散点图
fig=plt.figure()
ax=fig.add_subplot(121)
ax.scatter(return_mat[:,0],return_mat[:,1])
ax.set_xlabel('每年获得的飞行常客里程数')
ax.set_ylabel('玩视频游戏所耗时间百分百')

ax=fig.add_subplot(122)
ax.scatter(return_mat[:,0],return_mat[:,1],15.0*array(label_vec),15.0*array(label_vec))

plt.show()
'''

#3.归一化
def autoNorm(dataSet):
    minvals=dataSet.min(0)
    maxvals=dataSet.max(0)
    ranges=maxvals-minvals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0] #行数
    normDataSet=(dataSet-tile(minvals,(m,1)))/tile(ranges,(m,1)) #归一化公式
    return normDataSet,ranges,minvals
#输出
#normMat,ranges,minvals=autoNorm(return_mat)
#print(normMat,ranges,minvals)

#4.分类器
def ClassTest():
    hoRatio=0.10
    return_mat, label_vec=file2matrix('datingTestSet2.txt')
    normMat,ranges,minvals=autoNorm(return_mat)
    m=normMat.shape[0] #归一化后的数据行数(1000行)
    numTestVecs=int(m*hoRatio) #百分之十的数据(100行)用作测试集
    errorCount=0.0
    for i in range(numTestVecs):
        #输入向量集（测试向量），输入的训练样本集（剩余向量），标签向量（剩余向量对应标签），选择最近邻居的数目
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],label_vec[numTestVecs:m],3)
        print("the classifier came with: %d, the real answer is :%d " % (classifierResult, label_vec[i]))
        if(classifierResult != label_vec[i]) : errorCount += 1.0
    print("the error_rate is: %f " %(errorCount /float(numTestVecs)))
#测试分类器错误率
#ClassTest()

#5.约会网站预测函数
def classifyPerson():
    resultList=['not at all','in small doses','in large doses'] #输出结果可能列表
    #用户输入
    ffmiles=float(input('frequent flier miles earned per year?')) #再键入每年飞行里程
    percentTats=float(input('percentage of time wpent playing video games?')) #首先需要键入花在游戏上时间比重
    icecream=float(input('liters of ice cream consumed per year?')) #最后再键入在冰淇淋的消耗量
    #准备训练样本
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt') #把训练集文本数据转化为向量，便于后续处理
    normMat,ranges,minVals=autoNorm(datingDataMat) #几个关键数据量归一化，便于处理
    #分类
    inArr=np.array([ffmiles,percentTats,icecream]) #將刚刚键入的数，组成测试集
    classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3) #將预测结果存储
    print('You will probably like this guy:',resultList[classifierResult-1]) #给出预测结果
#分类结果
classifyPerson()