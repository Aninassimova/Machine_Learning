from numpy import * # 载入numpy库
import operator # 载入operator模块

def createDataSet(): # 建立数据集
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k): # 简单kNN分类器
    # 首先用欧式距离法计算已知类别数据集与当前点的距离
    dataSetSize=dataSet.shape[0] # 读取数据集的行数，并把行数放到dataSetSize里，shape[]用来读取矩阵的行列数，shape[1]表示读取列数
    diffMat=tile(inX,(dataSetSize,1))-dataSet # tile(inX,(dataSetSize,1))复制比较向量inX，tile的功能是告诉inX需要复制多少遍，这里复制成(dataSetSize行，1次)
                                              # 目的是把inX转化成与数据集相同大小，再与数据集矩阵相减，形成的差值矩阵存放在diffMat里
    sqDiffMat=diffMat**2 # 注意这里是把矩阵里的各个元素依次平方，如（[-1,-1.1],[-1,-1]）执行该操作后为（[1,1.21],[1,1]）
    sqDistances=sqDiffMat.sum(axis=1) # 实现计算计算结果，axis表矩阵每一行元素相加，如（[1,1.21],[1,1]）,执行该操作后为（2.21，2）
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

# 验证
group,labels=createDataSet()
classify_result=classify0([1,2],group,labels,3)
print(classify_result)