from math import log

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
        #计算每种划分方式的信息熵
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value) #划分出 第i列特征值为value的数据集（去除了第i列特征）
            prob=len(subDataSet)/float(len(dataSet)) #第i列特征取值为value的概率
            newEntropy += prob*calcShannonEnt(subDataSet) #按第i列特征划分的信息熵
        #计算最好的信息增益#將最信息增益先置零
        infoGain=baseEntropy-newEntropy #信息增益（熵的减少）=按类别划分的信息熵-按第i列特征划分的信息熵
        if infoGain>bestInfoGain: #如果信息增益比现有最好增益还大
            bestInfoGain=infoGain #则取代他
            bestFeature=i #并记下此时的分割位置
    return bestFeature #返回分割位置
#测试
print(chooseBestFeatureToSplit(myDat))