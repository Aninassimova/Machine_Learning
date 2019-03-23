''''
试验某些模块
'''
from numpy import * # 载入numpy库
import operator # 载入operator模块

a=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
sqDistances=a.sum(axis=1)
distances=sqDistances**0.5 # 开根号

# 假设欧氏距离直接试验k-均值法
distances=array([0.1,1.5,1.48,1.56])
sortedDisIndicies=distances.argsort()
print(sortedDisIndicies) # [0 2 1 3]

labels=['A','A','B','B']
classCount={} # 新建一个字典，用于计数

for i in range(3):
    voteIlabel=labels[sortedDisIndicies[i]] # 第i名对应的类别
    print(voteIlabel)
    print(classCount.get(voteIlabel,0)+1) # 计数:归类为A（B）的次数
    classCount[voteIlabel]=classCount.get(voteIlabel,0)+1 # 给字典赋值
print(classCount)
d=classCount.items() # 把字典转化为列表，每个元素是一个tuple，tuple的第一个元素是键，第二个元素是值
print(d)
sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True) #按位置1元素降序排列
print(sortedClassCount)
print(sortedClassCount[0][0]) # 返回出现次数最多到label值，即为当前点的预测分类