from numpy import * #载入numpy库
import numpy as np
import operator #载入operator模块
from os import listdir #从os模块导入listdir，可以给出给定目录文件名

#简单kNN分类器
def classify0(inX,dataSet,labels,k):
    # 首先用欧式距离法计算已知类别数据集与当前点的距离
    dataSetSize=dataSet.shape[0] # 读取数据集的行数，并把行数放到dataSetSize里，shape[]用来读取矩阵的行列数，shape[1]表示读取列数
    diffMat=tile(inX,(dataSetSize,1))-dataSet # tile(inX,(dataSetSize,1))复制比较向量inX，tile的功能是告诉inX需要复制多少遍，这里复制成(dataSetSize行，1次)。目的是把inX转化成与数据集相同大小，再与数据集矩阵相减，形成的差值矩阵存放在diffMat里
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
    return sortedClassCount[0][0] # 返回出现次数最多到label值，即为当前点的预测分类

#将文本转换为矩阵
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

#归一化
def autoNorm(dataSet):
    minvals=dataSet.min(0)
    maxvals=dataSet.max(0)
    ranges=maxvals-minvals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0] #行数
    normDataSet=(dataSet-tile(minvals,(m,1)))/tile(ranges,(m,1)) #归一化公式
    return normDataSet,ranges,minvals

#1.將图像转化为向量
def img2vector(filename):
    fr=open(filename) #打开文件
    returnVector=zeros((1,1024)) #构建预存一维向量，大小之所以为1024是因为图片大小为32*32=1*1024
    #將每行图像均转化为一维向量
    for i in range(32):
        lineStr=fr.readline() #按行读入每行数据
        for j in range(32):
          returnVector[0,32*i+j]=int(lineStr[j]) #將每行的每个数据依次存到一维向量中
    return returnVector #返回处理好的一维向量
#输出测试
#print(img2vector('digits/testDigits/0_1.txt'))

#2.使用k近邻算法识别手写数字
def handwritingClassTest():
    #获取训练集目录内容
    hwLabels = [] #训练集的标签矩阵
    trainingFileList = listdir('digits/trainingDigits') #返回trainingDigits目录下的文件名
    m = len(trainingFileList) #返回文件夹下文件的个数(1934)
    trainingMat = np.zeros((m, 1024)) #初始化训练的Mat矩阵,测试集向量大小为训练数据个数*1024，即多少张图像，就有多少行，一行存一个图像
    #从文件名中解析出训练集的类别标签
    for i in range(m):
        fileNameStr = trainingFileList[i] #获得文件的名字
        classNumber = int(fileNameStr.split('_')[0]) #第一个字符串存储标签，故取分离后的第一个元素，即相当于获取了该图像类别标签
        hwLabels.append(classNumber) #将获得的类别标签添加到hwLabels中
        trainingMat[i,:] = img2vector('digits/trainingDigits/%s' % (fileNameStr)) #将每一个文件的1x1024数据存储到trainingMat中

    #获取测试集目录内容
    testFileList = listdir('digits/trainingDigits') #返回testDigits目录下的文件列表
    errorCount = 0.0 #错误检测计数，初始值为0
    mTest = len(testFileList) #测试数据的数量(1934)
    #从文件名中解析出测试集的类别标签
    for i in range(mTest): #从文件中解析出测试集的类别并进行分类测试
        fileNameStr = testFileList[i]#获得文件的名字
        classNumber = int(fileNameStr.split('_')[0])#获得分类的数字标签
        vectorUnderTest = img2vector('digits/trainingDigits/%s' % (fileNameStr)) #获得测试集的1x1024向量
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)#获得预测结果
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if(classifierResult != classNumber):#如果预测结果与实际结果不符，则错误数加一
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount/mTest * 100))#获取错误率

#测试
handwritingClassTest()