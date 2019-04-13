import matplotlib.pyplot as plt

#1.使用文本注释绘制树节点
decisionNode = dict(boxstyle="sawtooth", fc="0.8") # boxstyle是文本框类型 fc是边框粗细 sawtooth是锯齿形
leafNode = dict(boxstyle="round4", fc="0.8") #有圆角的矩形框
arrow_args = dict(arrowstyle="<-") #箭头格式
#画节点
def plotNode(nodeTxt, centerPt, parentPt, nodeType): #nodeTxt用于记录nodeTxt，即节点的文本信息。centerPt表示那个节点框的位置。 parentPt表示那个箭头的起始位置。nodeType表示的是节点的类型。
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', # annotate 用以注释
    xytext=centerPt, textcoords='axes fraction',
    va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)
#画图
def createPlot():
    fig = plt.figure(1, facecolor='white')  #新建一个画布，背景设置为白色的
    fig.clf()  #将画图清空
    createPlot.ax1 = plt.subplot(111, frameon=False)  #设置一个多图展示，但是设置多图只有一个
    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()
#createPlot()

#2.获取叶节点的数目和树的层数。以叶子节点个数来确定X轴长度；以树的层数来确定y轴长度。
def getNumLeaves(myTree): #{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
    numLeaves = 0
    #Python3中使用LIST转换first_node，原书使用[0]直接索引只能用于Python2
    firstNode = myTree.keys() #dict_keys(['no surfacing'])
    firstNode = list(firstNode)[0] #no surfacing(list中元素——第一个特征)
    secondDict = myTree[firstNode] #secondDict={0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}
    #对于树，每次判断value是否为字典，若为字典则进行递归，否则累加器+1
    for key in secondDict.keys(): #secondDict.keys()=dict_keys([0, 1])
        if type(secondDict[key]).__name__ =='dict': #str——('no'),dict——({'flippers': {0: 'no', 1: 'yes'}})
            numLeaves += getNumLeaves(secondDict[key]) #出现一个新的判断节点，递归
        else: numLeaves += 1 #否则下面无子节点
    return numLeaves #1+（1+1）=3

def getTreeDepth(myTree):
    depth = 0
    firstNode = myTree.keys()
    firstNode = list(firstNode)[0]
    secondDict = myTree[firstNode]
    # 对于树，每次判断value是否为字典，若为字典则进行递归，否则计数器+1
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ =='dict':
            pri_depth = 1 + getTreeDepth(secondDict[key]) #一旦遇到叶子节点，则从递归调用中返回，并将深度加1
        else: pri_depth = 1
        if pri_depth > depth: depth = pri_depth
    return depth #1>0取1--->(1+1)>1取2

#预创建树
def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]
#测试
'''
myTree=retrieveTree(0)
print(getNumLeaves(myTree)) #3
print(getTreeDepth(myTree)) #2
'''

#3.构造注解树
def plotMidtext(cntrPt,parentPt,txtString):
    #作用是计算tree的中间位置。cntrPt起始坐标(子节点坐标),parentPt结束位置(父节点坐标),txtString：文本标签信息。
    xmid=(parentPt[0]-cntrPt[0])/2.0+cntrPt[0]
    ymid=(parentPt[1]-cntrPt[1])/2.0+cntrPt[1] #找到x和y的中间位置
    createPlot.ax1.text(xmid,ymid,txtString) #在父子节点间填充文本信息

def plotTree(myTree,parentPt,nodeTxt):
    #(1)绘制自身  (2)判断子节点非叶子节点，递归  (3)判断子节点为叶子节点，绘制
    numLeafs = getNumLeaves(myTree)  #叶节点数（宽）
    depth = getTreeDepth(myTree) #树的层数（高）
    firstStr=list(myTree.keys())[0] #取第一个key（特征名）
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff) #计算子节点的坐标,xOff=当前偏移量+初始偏移量+当前节点相对位置（当前节点的叶子节点所占的总距离/2）
    plotMidtext(cntrPt, parentPt, nodeTxt) #绘制线上的文字
    plotNode(firstStr, cntrPt, parentPt, decisionNode) #绘制节点(锯齿形文本框，内填特征名)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD #每绘制一次图，将y的坐标减少1.0/plotTree.totalD，间接保证y坐标上深度
    #对于树，每次判断value是否为字典，若为字典则进行递归
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode) #绘制叶子节点(圆角矩形文本框，内填'yes'或'no')
            plotMidtext((plotTree.xOff, plotTree.yOff), cntrPt, str(key)) #绘制线上的文字，str(key)为0或1
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
    #类似于Matlab的figure，定义一个画布长1.0、高1.0，背景为白色
    fig=plt.figure(1,facecolor='white')
    fig.clf()    #把画布清空
    axprops = dict(xticks=[], yticks=[])
    #createPlot.ax1为全局变量，绘制图像的句柄，subplot为定义了一个绘图，
    #111表示figure中的图有1行1列，即1个，最后的1代表第一个图
    #frameon表示是否绘制坐标轴矩形
    createPlot.ax1 = plt.subplot(111, frameon=True,**axprops)
    plotTree.totalW = float(getNumLeaves(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    #在开始的时候plotTree.xOff的赋值为-0.5/plotTree.totalW,即意为开始x位置为第一个表格左边的半个表格距离位置，这样作的好处为：在以后确定叶子节点位置时可以直接加整数倍的1/plotTree.totalW
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0; #totalW为整树的叶子节点树，totalD为深度
    plotTree(inTree, (0.5,1.0), '') #开始的根节点不用划线，因此父节点和当前节点的位置需要重合，当前节点的位置便为(0.5, 1.0)
    plt.show()
#测试
#myTree=retrieveTree(0)
#createPlot(myTree)