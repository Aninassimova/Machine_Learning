#Python中字符串相关的处理都非常方便，来看例子：
a = 'Life is short, you need Python'
a.lower()              	# 'life is short, you need Python'
a.upper()               	# 'LIFE IS SHORT, YOU NEED PYTHON'
a.count('i')            	# 2
a.find('e')             	# 从左向右查找'e'，3
a.rfind('need')         	# 从右向左查找'need'，19
a.replace('you', 'I')   # 'Life is short, I need Python'
tokens = a.split()    	# ['Life', 'is', 'short,', 'you', 'need', 'Python']
print(tokens)
b = ' '.join(tokens)	# 用指定分隔符按顺序把字符串列表组合成新字符串
print(b)
c = a + '\n'            	# 加了换行符，注意+用法是字符串作为序列的用法
c.rstrip()              	# 右侧去除换行符
d=[x for x in a]          	# 遍历每个字符并生成由所有字符按顺序构成的列表
print(d)
'Python' in a   			# True

#类（Class）
#Python中的类的概念和其他语言相比没什么不同，比较特殊的是protected和private在Python中是没有明确限制的，一个惯例是用单下划线开头的表示protected，用双下划线开头的表示private：
'''
class A:
    """Class A"""
    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self._name = name

    def introduce(self):
        print(self._name)

    def greeting(self):
        print("What's up!")

    def __l2norm(self):
        return self.x**2 + self.y**2

    def cal_l2norm(self):
        return self.__l2norm()

a = A(11, 11, 'Leonardo')
print(A.__doc__)        	# "Class A"
a.introduce()           	# "Leonardo"
a.greeting()            	# "What's up!"
print(a._name)          	# 可以正常访问
print(a.cal_l2norm())   # 输出11*11+11*11=242
print(a._A__l2norm())   # 仍然可以访问，只是名字不一样
print(a.__l2norm())     	# 报错: 'A' object has no attribute '__l2norm'
'''

#Python中的继承也非常简单，最基本的继承方式就是定义类的时候把父类往括号里一放就行了：
'''
class B(A):
    """Class B inheritenced from A"""
    def greeting(self):
        print("How's going!")

b = B(12, 12, 'Flaubert')
b.introduce()   # Flaubert
b.greeting()    # How's going!
print(b._name())        # Flaubert
'''