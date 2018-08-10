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


############## 数据归一化要加的代码 ###########################
#特征变量归一化
def autoNorm(dataSet):
    
    #取出每一列的最小值，即每一个特征的最小值
    minVals = dataSet.min(0)
    
    #取出每一列的最大值，即每一个特征的最大值
    maxVals = dataSet.max(0)
    
    #每一个特征变量的变化范围
    ranges = maxVals - minVals
    
    #初始化待返回的归一化特征数据集
    normDataSet = zeros(shape(dataSet))
    
    #特征数据集行数，即样本个数
    m = dataSet.shape[0]
    
    #利用tile()函数构造与原特征数据集同大小的矩阵，并进行归一化计算
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    
    return normDataSet,ranges,minVals


############## 测试算法要加的代码 #####################
#分类器测试
def datingClassTest():
    
    #验证集占据整个训练集的比例
    hoRatio = 0.10
    
    #解析文本,分别分开特征和标签
    datingDataMat,datingLabels = file2matrix('F:/programming tools/datingTestSet2.txt')
    
    #归一化
    normMat,ranges,minVals = autoNorm(datingDataMat)
    
    #训练集个数
    m = normMat.shape[0]
    
    #验证集个数
    numTestVecs = int(m*hoRatio)
    
    #初始化分类器犯错样本个数
    errorCount = 0.0
    
    for i in range(numTestVecs):
        # 90%训练集输入作为训练样本，对验证集进行分类
        # classify0(验证集特征,训练集特征,训练集标签,距离)
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        
        #对比分类器对测试样本预测的类别和其真实类别
        print("the classifier came back with: %d,the real answer is: %d" % (classifierResult,datingLabels[i]))
        
        #统计分类出错的测试样本数
        if (classifierResult != datingLabels[i]):errorCount+=1.0

    #输出分类器错误率
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)

############ 约会网站预测函数  #########################
def classifyPerson():
    
    #定义一个存储了三个字符串的列表，分别对应不喜欢，一般喜欢，很喜欢
    resultList = ['not at all','in small dose','in large dose']
    
    #用户输入三个特征变量，并将输入的字符串类型转化为浮点型
    ffMiles = float(input("frequent flier miles earned per year:"))
    percentats = float(input("percentage of time spent playing video games:"))
    iceCream = float(input("liters of ice cream consumed per year:"))
    
    #调用文本数据解析函数
    datingDataMat,datingLabels = file2matrix('/Users/bindo/Desktop/datingTestSet2.txt')
    
    #调用特征变量归一化函数
    normMat,ranges,minVals = autoNorm(datingDataMat)
    
    #将输入的特征变量构造成特征数组（矩阵）形式
    inArr = array([ffMiles,percentats,iceCream])
    
    #调用kNN简单实现函数
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    
    #将分类结果由数字转换为字符串
    print("You will probably like this person",resultList[classifierResult - 1])


############### 手写系统把数据处理成分类器需要格式的函数 ##########################
def img2vector(filename):
    
    # numpy.zeros((n,m)) 创建一个n行m列的零矩阵，准备接收后面返回的数值
    # 这里创建一个1行1024列的，零矩阵
    returnVect = numpy.zeros((1,1024))
    
    # 打开文件
    fr = open(filename)
    
    # 我们目的是循环读出文件的前32行
    for i in range(32):
        
        # 一行一行读取文件,返回list []
        lineStr = fr.readline()
            
        # 读取每一行的前32个字符
        for j in range(32):
                
            # 零矩阵的第1行第33列 放入 第1个list的第1个数值
            # 零矩阵的第1行第34列 放入 第1个list的第2个数值
            # 到这里我还想不清楚为什么要这么弄
            returnVect[0,32*i+j] = int(lineStr[j])

    # 最后返回处理好的矩阵
    return returnVect

################## 手写识别的测试代码3############################################
def handwritingClassTest():

    # 声明hwLabels是一个列表
    hwLabels = []

    # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
    # 这里是要返回所有在trainingDigits的txt文件
    trainingFileList = os.listdir('/Users/bindo/Desktop/Machine Learning/machinelearninginaction/Ch02/digits/trainingDigits')

    # 看一下有多少个txt文件
    m = len(trainingFileList)

    # 创建m行1024列的零矩阵
    # 我们想要把每个txt文件都弄成是1行1024列的矩阵，这样就可以把所有文件集成一个大矩阵
    trainingMat = numpy.zeros((m,1024))

    # 循环每一个txt文件
    for i in range(m):

        # 拿到第i个文件
        fileNameStr = trainingFileList[i]

        # 比如上面拿到第1个文件，叫做0_0.txt，我们把.当做分隔符分开，得到0_0和txt,然后选择第一个的，也就是剩下0_0
        fileStr = fileNameStr.split('.')[0]

        # 上面处理完了，剩下0_0，我们把_当做分隔符，又分开得到0和1，然后选择第一个的，也就是0，我们还让0变成int的0才输出
        classNumStr = int(fileStr.split('_')[0])

        # 把所有文件都这么处理后，剩下的那个int的数，都放到最初声明的hwLabels列表中
        hwLabels.append(classNumStr)

        # 调用上面的img2vector(filename)函数，最后img2vector返回的每一个文件变成一个处理好的矩阵后，
        # 逐个按照i的顺序插入到零矩阵中
        trainingMat[i,:] = img2vector('/Users/bindo/Desktop/Machine Learning/machinelearninginaction/Ch02/digits/trainingDigits/%s' % fileNameStr)

    # 我们返回所有在testDigits的txt文件
    testFileList = os.listdir('/Users/bindo/Desktop/Machine Learning/machinelearninginaction/Ch02/digits/testDigits')

    # 初始化分类器犯错样本个数
    errorCount = 0.0

    # 看一下测试集有多少个txt文件
    mTest = len(testFileList)

    # 遍历所有测试集的txt文件
    for i in range(mTest):

        # 拿到测试集的第i个文件
        fileNameStr = testFileList[i]

        # 和上面一样的分隔符处理取整数输出值
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])

        # 调用img2vector()对每一个测试集的txt文件处理输入一个1行1024列的矩阵
        # 然后按照i的顺序逐个插入到mTest行1024列的零矩阵中
        vectorUnderTest = img2vector('/Users/bindo/Desktop/Machine Learning/machinelearninginaction/Ch02/digits/testDigits/%s' % fileNameStr)

        # 调用classify0()分类器函数，
        #classify0(测试集特征，训练集特征，训练集标签，k)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)

        # 把预测结果和真实结果对比一下
        print("the classifier came back with %d, the real answer is: %d" % (classifierResult,classNumStr))

        # 进行判断，如果测试结果和真实的测试集结果不一样，那么errorCount就加1
        if (classifierResult != classNumStr): errorCount += 1.0

    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))






