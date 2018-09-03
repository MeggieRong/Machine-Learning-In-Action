
import numpy as np
########## 程序清单5-1 Logistic回归梯度上升优化算法 ###########################

#loadDataSet()用于打开txt，然后逐行读取
#每行前两个值分别是X1,和X2，第三个值是类别标签
def loadDataSet():
    dataMat,labelMat = [],[]
    with open('/Users/bindo/Desktop/Machine Learning/machinelearninginaction/Ch05/testSet.txt',"r") as  fr:  #open file
        for line in fr.readlines():
            lineArr = line.split() #split each line
            dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])  #创建2维list
            labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

#弄一下sigmoid的计算
def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMat,labelMat):
    dataMatrix = np.mat(dataMat)  #translate list to matrix
    labelMatrix = np.mat(labelMat).transpose() #转置
    m,n = np.shape(dataMatrix) #100 rows  3 coulums
    alpha = 0.001 #步长 or 学习率
    maxCyclse = 500
    weight = np.ones((n,1)) #初始值随机更好吧
    #weight = np.random.rand(n,1)
    for k in range(maxCyclse):
        h = sigmoid(dataMatrix * weight) # h 是向量
        error = (labelMatrix - h)  #error 向量
        weight = weight + alpha * dataMatrix.transpose() *error  #更新
    #   print(k,"  ",weight)
    return weight


########### 程序清单5-2 画出数据集和Logistic回归最佳拟合直线的函数#################
def plotfit(wei):
    import matplotlib.pyplot as plt
    weight = np.array(wei) #???????? #return array
    dataMat ,labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]  #row
    fig = plt.figure()   #plot
    ax = fig.add_subplot(111)
    ax.scatter(dataArr[:,1],dataArr[:,2],s =50, c = np.array(labelMat)+5) #散点图 #参考KNN 的画图
    x = np.arange(-3.0,3.0,0.1)   #画拟合图像
    y = (-weight[0] - weight[1] *x ) / weight[2]
    ax.plot(x,y)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


#######程序清单5-3随机梯度上升算法 #########################################
def stocGradAscent0(dataMatrix, classLabels):
    m,n =shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

###### 程序清单5-4 改进的随机梯度上升算法 #################################3
def stocGradAscent1(dataMat,labelMat,numIter = 150):
    dataMatrix = np.mat(dataMat)  #translate list to matrix
    labelMatrix = np.mat(labelMat).transpose() #转置
    m,n = np.shape(dataMat)
    alpha = 0.1
    weight = np.ones(n) #float
    #weight = np.random.rand(n)
    for j in range(numIter):
        dataIndex = list(range(m)) #range 没有del 这个函数　　所以转成list  del 见本函数倒数第二行
        for i in range(m):
            alpha = 4/(1.0 +j + i) + 0.01
            randIndex = int(np.random.uniform(0,len(dataIndex))) #random.uniform(0,5) 生成0-5之间的随机数
            #生成随机的样本来更新权重。
            h = sigmoid(sum(dataMat[randIndex] * weight))
            error = labelMat[randIndex] - h
            weight = weight + alpha * error * np.array(dataMat[randIndex])  #!!!!一定要转成array才行
            #dataMat[randIndex] 原来是list  list *2 是在原来的基础上长度变为原来2倍，
            del(dataIndex[randIndex]) #从随机list中删除这个
    return weight


################## 程序清单5-5 Logistic回归分类函数 ###########################

def classifyVector(inX,weight):  #输入测试带测试的向量 返回类别
    prob = sigmoid(sum(inX * weight))
    if prob > 0.5 :
        return 1.0
    else: return 0.0
def colicTest():
    trainingSet ,trainingSetlabels =[],[]
    with open("/Users/bindo/Desktop/Machine Learning/machinelearninginaction/Ch05/horseColicTraining.txt") as frTrain:
        for lines in frTrain.readlines():
            currtline = lines.strip().split('\t')  # strip()remove the last string('/n') in everyline
            linearr = [] #每行临时保存str 转换float的list
            for i in range(21):   #将读进来的每行的前21个str 转换为float
                linearr.append(float(currtline[i]))
            trainingSet.append(linearr)  #tianset 是2维的list
            trainingSetlabels.append(float(currtline[21]))#第22个是类别
    trainWeights = stocGradAscent1(trainingSet,trainingSetlabels,500)
    errorCount = 0
    numTestVec = 0.0
    with open("/Users/bindo/Desktop/Machine Learning/machinelearninginaction/Ch05/horseColicTest.txt") as frTrain:
        for lines in frTrain.readlines():
            numTestVec += 1.0
            currtline = lines.strip().split('\t')  # strip()remove the last string('/n') in everyline
            linearr = []  #测试集的每一行
            for i in range(21):
                linearr.append(float(currtline[i]))#转换为float
            if int(classifyVector(np.array(linearr),trainWeights)) != int(currtline[21]) :
                errorCount += 1  #输入带分类的向量，输出类别，类别不对，errorCount ++
            errorRate = float(errorCount)/numTestVec
            print("the error rate of this test is : %f"%errorRate)
    return errorRate
def multiTest(): #所有测试集的错误率
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum +=colicTest()
    print("after %d iterations the average error rate is : %f" %(numTests,errorSum/float(numTests)))
