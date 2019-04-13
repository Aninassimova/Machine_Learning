from math import log
import operator

#1.计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries=len(dataSet)
    labelCounts={}
    #为所有可能的分类创建字典
    for featVec in dataSet:
        currentLabel=featVec[-1] #键值为最后一列的数值
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0 #遇到未统计过的类别，类别作为key加入字典，值初始化为0
        labelCounts[currentLabel] += 1 #统计类别（currentLabel）发生次数  labelCounts={'yes': 3, 'no': 2}
    #求香农熵
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries #该类别下发生概率
        shannonEnt -= prob*log(prob,2) #以2为底数求对数
    return shannonEnt
#测试
def createDataSet():#建立数据集
    dataSet=[[1,1,'yes'],
        [1,0,'no'],
        [0,1,'no'],
        [1,1,'yes'],
        [0,0,'yes']]
    labels=['no surfacing','flippers']
    return dataSet,labels
myDat,labels=createDataSet()
#print(calcShannonEnt(myDat))

#2.划分数据集
#按照给定特征划分数据集
def splitDataSet(dataSet,axis,value): #待划分数据集，划分数据集的特征列，需要返回的特征的值
    retDataSet=[] #构建新list
    #对于数据集，抽取其中符合特征的数据
    for featVec in dataSet:
        if featVec[axis]==value: #一旦发现特征与目标值一致
            reducedFeatVec=featVec[:axis] #將axis之前的列附到resucdFeatVec里
            reducedFeatVec.extend(featVec[axis+1:]) #将axis以后的列附到reducedFeatVec里
            retDataSet.append(reducedFeatVec) #將去除了指定特征列的数据集放在retDataSet里
    return retDataSet #返回划分后数据集
#print(splitDataSet(myDat,0,0)) #[[1, 'no'], [0, 'yes']]

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeature=len(dataSet[0])-1 #特征数量=数据列长-1（类别）
    baseEntropy=calcShannonEnt(dataSet) #按类别（最后一列）分类的信息熵
    bestInfoGain=0.0 #將最好信息增益先置零
    bestFeature=-1 #將最佳分割点值-1，不置0是因为0相当于第一个分割点，会引起误解
    for i in range(numFeature):
        #创建唯一的分类标签列表
        featList=[example[i] for example in dataSet] #循环取数据集的第i列特征
        uniqueVals=set(featList) #去重，提取第i列特征值
        newEntropy=0.0
        #计算每种划分方式的信息熵 公式：H(Y|X)=P(X=0)H(Y|X=0)+P(X=1)H(Y|X=1)——X为特征，这里的数据集中假设它有0和1两个取值；Y为类别，即yes和no
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value) #划分出 第i列特征值为value的数据集（去除了第i列特征）
            prob=len(subDataSet)/float(len(dataSet)) #第i列特征取值为value的概率
            newEntropy += prob*calcShannonEnt(subDataSet) #按第i列特征划分的信息熵
        #计算最好的信息增益
        infoGain=baseEntropy-newEntropy #信息增益（熵的减少）=按类别划分的信息熵-按第i列特征划分的信息熵
        if infoGain>bestInfoGain: #如果信息增益比现有最好增益还大
            bestInfoGain=infoGain #则取代他
            bestFeature=i #并记下此时的分割位置
    return bestFeature #返回分割位置
#测试
#print(chooseBestFeatureToSplit(myDat)) #0

#3.递归构建决策树
#投票表决
def majorityCnt(classList):
    classCount={} #建立一个数据字典，里面存储所有的类别
    for vote in classList:
        if vote not in classList.key():classCount[vote]=0 #如果有新的类别，则创立一个新的元素代表该种类
        classCount[vote] += 1 #否则该元素加一
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True) #对数据集进行排序，第二列作为排序依据，从高到低排
    return  sortedClassCount[0][0] #把第一个元素返回，即返回出现次数最多的那个元素

#创建树
def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet] #以数据集的最后一列作为新的一个列表
    if classList.count(classList[0])==len(classList): #如果分类列表完全相同
        return classList[0] #停止继续划分
    if len(dataSet[0])==1: #如果遍历完所有特征，仍不能划分为唯一门类
        return majorityCnt(classList) #返回出现出现次数最多的那个类标签
    bestFeat=chooseBestFeatureToSplit(dataSet) #否则选择最优特征
    bestFeatLabel=labels[bestFeat] #同时將最优特征的标签赋予bestFeatureLabel
    myTree = {bestFeatLabel:{}} #根据最优标签生成树
    del(labels[bestFeat]) #將刚刚生成树所使用的标签去掉
    featValues=[example[bestFeat] for example in dataSet] #获取所有训练集中最优特征属性值
    uniqueVals=set(featValues) #把重复的属性去掉，并放到uniqueVals里
    for value in uniqueVals: #遍历特征的所有属性值
        subLabels=labels[:] #先把原始标签数据完全复制，防止对原列表干扰
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels) #再以该特征来划分决策树
    return myTree #返回决策树
#测试
#print(createTree(myDat,labels))

#4.使用决策树执行分类
def classify(inputTree,featLabels,testVec):# inputTree:决策树   featLabels:测试数据标签['no surfacing', 'flippers']   testVec:测试数据值[1,0]
    firstStr=list(inputTree.keys())[0] #no surfacing--flippers
    secondDict=inputTree[firstStr] #{0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}--{0: 'no', 1: 'yes'}
    featIndex=featLabels.index(firstStr) #将标签字符串转换为索引 0--1
    for key in secondDict.keys(): #dict_keys([0, 1])
        if testVec[featIndex]==key: #testVec[featIndex]为当前最优特征的值，1表示有该特征，0表示无该特征
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabels,testVec)
            else:   classLabel=secondDict[key]
    return classLabel
#测试
def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]
myTree=retrieveTree(0)
#print(classify(myTree,labels,[1,0]))
#print(classify(myTree,labels,[1,1]))

#5.因为构建决策树耗时严重， 因此构建成功将决策树保存，然后测试时从文件中直接读取使用
#将构建的决策树写入文件
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'wb') #二进制写
    pickle.dump(inputTree,fw,0) #0：ASCII协议，所序列化的对象使用可打印的ASCII码表示；
                                #1：老式的二进制协议；
                                #2：2.3版本引入的新二进制协议，较以前的更高效。其中协议0和1兼容老版本的python。protocol默认值为3。
    fw.close()

#从文件中读取决策树
def grabTree(filename):
    import pickle
    fr = open(filename,'rb') #二进制读
    return pickle.load(fr)
#测试
#storeTree(myTree,'ClassifierStorage.txt')
#print(grabTree('ClassifierStorage.txt'))