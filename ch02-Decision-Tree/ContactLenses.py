from trees import createTree
from treePlotter import createPlot
from numpy import * #载入numpy库

# 加载隐形眼镜数据集，并将其序列化，最后生成决策树
fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
print(lenses)
# 年龄、症状、是否散光、眼泪数量四个属性
lensesLabels = ['age','prescript','astigmatic','tearRate','type']
# 根据隐形眼镜的数据集和属性标签构造决策树
lensesTree = createTree(lenses,lensesLabels) #生成树
print(lensesTree)
createPlot(lensesTree) #绘制树