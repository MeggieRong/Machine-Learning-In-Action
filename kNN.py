# 初始自定义kNN模块
from numpy import * # 导入科学计算包
import operator     # 导入python函数库，运算符模块

# 创建数据集
def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group, labels 

# 自定义classify0()函数
# classify0()函数需要在时候的我们提供四个参数，分别是
# inX ：目标未知向量，是需要我们进行分类判断的向量
# dataSet : 训练集
# labels : 训练集的特征，也就是标签向量
def classify0(inX, dataSet, labels, k):
    # 一、先算距离
    
    # 得到训练集的行数
    dataSetSize = dataSet.shape[0]
    
    # tile是numpy的函数，原型为 numpy.tile(A,reps)，一共两个参数
    # A是待输入数组，reps是决定A重复的次数
    # 下面的待输入数组为未知向量inX,需要未知向量竖向重复行数次，这样就可以和每一个已知的向量进行相减
    # 如果不能理解，可以一行一行运行下面的代码，看到结果就明白了
    # from numpy import *
    # a = array([[1,1.1],[1,1],[0,0],[0,0.1]])
    # a
    # a.shape
    # a.shape[0]
    # b = tile(a,3)
    # b
    #c = tile(a,(3,1)) #这里跟着3在括号里的1，是需要纵向重复的意思，平时不写，或者写0，都是横向的意思
    # c
    
    # 所以下面我们就能得到目标和训练数值之间的差值
    diffMat = tile(inX, (dataSetSize,1)) - dataSet

    # 这么理解,每个点都有横坐标和纵坐标,这里是横坐标之差和纵坐标之差都让它们平方
    sqDiffMat = diffMat**2
    
    # 然后按照对应列,把横坐标之差的平方+纵坐标之差的平方
    sqDistances = sqDiffMat.sum(axis = 1)
    
    # 把上面的家和给开方,就是点与点之间的距离
    distances = sqDistances**0.5
    
    # 我们升序排序一下,把距离小的往前面排,sortedDistIndicies最后的值是 距离值
    sortedDistIndicies = distances.argsort()

    # 二、选择距离最小的k个点
    classCount = {} #先声明,classCount是一个字典  ##补充 列表list [] ; 元组tuple () ; 字典dict {} ; 集合set ()
    
    # range返回一个数组,i是遍历的每个数组的值 #可以尝试在IDLE输入 for i in range(10): 回车 print(i) ,可以看到效果
    for i in range(k):
        
        # 这里是选择距离最小的k个点， sortedDistIndicies已经排好序，只需迭代的取前k个样本点的labels(即标签)
        voteIlabel = labels[sortedDistIndicies[i]]
        
        # 拿到上面的标签之后,统计该标签出现的次数
        # 这里用到dict.get(key, default=None)函数
        # dict字典就是刚开始声明的classCount, key就是标签, 我们找到这个标签就+1,没有找到就返回0
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1

    # 使用sorted()函数, 函数格式为sorted(iterable, cmp=None, key=None, reverse=False)
    # iterable是可迭代类型;key用列表元素的某个属性或函数进行作为关键字;reverse=True降序/False升序,默认是降序
    # classCount.items() 将classCount字典分解为元组列表
    # operator.itemgetter(1) 获取对象第一个域的值，在python里面的排序，是0，1，2，3，4，所以这里选择1，意思是按照标签的次数来排序
    #这个语句作用是把字典形式的{标签1:次数,标签2:次数,标签3:次数...}分开为列表
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)
    
    #sortedClassCount[0]是标签数最大的那个元组，再多加一个0，就是返回这个元组的标签名称
    return sortedClassCount[0][0]




############## 约会事例要加的代码 ##############################
import numpy
def file2matrix(filename):
    fr = open(filename) #打开文件
    arrayOLines = fr.readlines() #一行一行读取文件,返回list []
    numberOfLines = len(arrayOLines) #获得行数,我们偷偷看了一下,是1000行
    returnMat = numpy.zeros([numberOfLines,3]) #创建一个1000行,3列的0矩阵(因为有3个特征),没错整个矩阵都是0,为了后面可以插入数据
    classLabelVector = [] #声明是一个列表,准备用来装返回的标签向量
    index = 0 #初始设置索引为0
    for line in arrayOLines :
        line = line.strip() #移除字符串头尾空格或换行符
        listFromLine = line.split('\t') #以'\t'作为分隔符,分割字符串
        returnMat[index,:] = listFromLine[0:3] #把每一行的三个特征遍历放到每一行的0矩阵中,最后得到一个数组
        classLabelVector.append(int(listFromLine[-1])) #append是增加的意思,classLabelVector是一个空列表,我们把listFromLine最后一列的数据加进去,让其格式为int
        index += 1 # 把索引从0开始遍历,逐个增加1,最后索引变成 0 1 2 3 4 ... 999
    return returnMat,classLabelVector #返回处理好的特征数组以及标签列表





















