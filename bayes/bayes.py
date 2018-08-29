
from numpy import *

############ 程序清单4-1 词表到向量的转换函数 ##################
#会创建一些实验样本
def loadDataSet():
    postingList = [
                   ['my','dog','has','flea','problems','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']
                    ]
    classVec = [0,1,0,1,0,1]
    return postingList, classVec
#会创建一个包含在所有文档中的不重复词的列表
def createVocabList(dataSet):
    vocabSet = set([]) #创建一个空集合
    for document in dataSet: #将每篇文档返回的新词集合添加到该集合中
        vocabSet = vocabSet | set(document) #操作符|用于求两个集合的并集
    return list(vocabSet)

#该函数的输入参数为词汇表及某个文档，输出的是文档向量，向量的每一个元素为1或者0，分别表示词汇表中的单词在输入文档中是否出现
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList) #首先创建一个和词汇表等长的向量，并将其元素都设置为0
    for word in inputSet: #接着遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出文档中对应值设为1
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else :  print ("the word %s is not in my Vocabulary!"%word)
    return returnVec

########## 程序清单 4-2 朴素贝叶斯分类器训练函数 ##################
#函数需要输入文档矩阵trainMatrix，以及由每一篇文档类别标签所构成的向量trainCategory
def trainNB0(trainMatrix,trainCategory) :
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords); p1Num = ones(numWords)
    p0Denom = 2.0 ; p1Denom = 2.0  #为了计算，P(wi|c1) 和 P(wi|C0)  到这里，初始化程序中的鞭自变量和分母变量
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i]) #到这里，在for循环中，遍历训练集trainMatrix中的所有文档，一旦某个词在某一文档中出现，则该词对应的个数（p1Num或者p0Num）就加1，而且在所有的文档中，该文档的总词数也相应加1
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom) #到这里就是每个元素除以该类别中的总词数
    return p0Vect,p1Vect,pAbusive


######## 程序清单 4-3 朴素贝叶斯分类函数 ##########################
#classifyNB的参数要输入的是 分类向量vec2Classify、以及使用函数trainNB0计算得到的三个概率
#这个函数的意义在于，使用Numpy的数组来计算两个向量相乘的结果，即对应元素相乘
#先将两个向量中的第1个元素相乘，然后将第2个元素相乘，以此类推
#然后再将词表中所有词的对应值相加
#最后，比较类别的概率返回大概率对应的类标签
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0 :
        return 1
    else:
        return 0


#这个testingNB()是一个封装函数，把所有操作都封装了，可以节省输入时间ce
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as :',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as :',classifyNB(thisDoc,p0V,p1V,pAb))


################ 程序清单4-4 词袋模型 ###########################
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


############### 程序清单 4-5 文本解析及完整的垃圾邮件测试函数 #######

#函数textParse()接受了一个大写字符串并将其解析为字符串列表，去掉了少于两个字符的字符串，并将所有字符串转为小写
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

#spamTest() 这个就是贝叶斯垃圾邮件分类器进行自动化处理
def spamTest():
    docList = []; classList = []; fullText = []
    
    #这一段for导入了文件夹spam和ham下的文本文件，并用textParse做好了解析
    for i in range(1,26):
        wordList = textParse(open('/Users/bindo/Desktop/Machine Learning/machinelearninginaction/Ch04/email 2/spam/%d.txt'%i,'r',encoding='UTF-8',errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        wordList = textParse(open('/Users/bindo/Desktop/Machine Learning/machinelearninginaction/Ch04/email 2/ham/%d.txt'%i,'r',encoding='UTF-8',errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50)); testSet = []


    #这一段for主要是随机构建训练集
    #两个集合中的邮件都是随机选出的，本例子中共有25+25=50个电子邮件，并不是很多，其中随机选10封电子邮件作为测试集
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []

    #这一段for主要是对测试集分类
    #选择出的数字所对应的文档被添加到测试列表，同时也将其从训练集中剔除
    #这种随机选择数据的一部分作为训练集，剩余部分作为测试集的过程称为留存交叉验证(hold-out cross validation)
    #假设现在只完成了一次迭代，那么为了更精确地估计分类器的错误率，应该进行多次迭代后求出平均错误率
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0

    #这一段for主要对测试集进行分类了
    #for循环遍历训练集的所有文档，对每个词基于词表且使用setOfWords2Vec()来构建词向量
    #这些词在trainNB0()用于计算分类所需的概率
    #然后遍历测试集，对其中每封电子邮件进行分类
    #如果邮件分类错误，则错误数+1，最后给出总的错误百分比
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print('The error rate is: ',float(errorCount)/len(testSet))


