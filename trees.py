from math import log # Math.log(number) 如果 number 为正，则此函数返回该数字的自然对数。如果 number 为负，则函数返回 NaN。如果 number 为 0，则此功能返回 -∞

import operator # 本模块主要包括一些Python内部操作符对应的函数。这些函数主要分为几类：对象比较、逻辑比较、算术运算和序列操作

def calcShannonEnt(dataSet):
    #dataset 为list  并且里面每一个list的最后一个元素为label
    # 如[[1,1,'yes'],
    #    [1,1,'yes'],
    #    [1,0,'no'],
    #    [0,0,'no'],
    #    [0,1,'no']]
    
    # 获得list的长度 即实例总数
    numEntried = len(dataSet)
    
    # 创建一个字典，来存储数据集合中不同label的数量 如 dataset包含3 个‘yes’  2个‘no’ （用键-值对来存储）
    labelCounts = {}

    # 对上面数据集的每一个样本进行for遍历
    for featVec in dataSet:
        
        #获得list里面每一个子list的最后一个位置的内容，也就是每个子list的label值
        currentLabel = featVec[-1]

        # 如果当前标签在字典键值中不存在
        if currentLabel not in labelCounts.keys():labelCounts[currentLabel] = 0
        #若已经存在 该键所对应的值加1
        labelCounts[currentLabel] += 1
    # 初值熵值为0.0
    ShannonEnt = 0.0

    #对于每一个label
    for key in labelCounts:
        # 概率probability，也就是这个分类出现的次数除以总共的分类数量
        prob = float(labelCounts[key])/numEntried
        # -= 减法赋值运算符 c -= a 等效于 c = c - a
        ShannonEnt -= prob * log(prob,2)
    return ShannonEnt


############# 利用createDataSet()简单鉴定鱼数据集#################
def createDataSet():
        dataSet = [[1, 1, 'yes'],
                   [1, 1, 'yes'],
                   [1, 0, 'no'],
                   [0, 1, 'no'],
                   [0, 1, 'no']]
        labels = ['no surfacing','filippers']
        return dataSet, labels



##############按照给定特征划分数据集##############################
# 定义划分数据集函数
#参数：待划分的数据集、划分数据集的列、划分数据集的列的对应值
def splitDataSet(dataSet, axis, value):
    #声明retDataSet 是一个列表
    retDataSet = []
    
    # 遍历每个子list
    for featVec in dataSet:
        # 如果第axis列的值是某个特征value
        if featVec[axis] == value:
            
            #featVec[：axis] 返回的是一个列表，其元素是featVec这个列表的索引从0到axis-1的元素
            # 也就是不包括axis这个索引上的值，若axis为0，则返回空列表
            reducedFeatVec = featVec[:axis]
            
            # 其中featVec[axis + 1: ]返回的是一个列表，其元素是featVec这个列表的索引从axis + 1开始的所有元素
            #featVec[:axis]和featVec[axis+1 :]组合起来了，就是要把axis这一列剔除掉，因为这一列是某个特征所在的列
            # 把抽取出该特征以后的所有特征组成一个列表
            reducedFeatVec.extend(featVec[axis+1 :])
            
            # 补充：方法extend和append的区别：
            #例子：
            #
            #    >>>a = [1, 2, 4]
            #
            #>>>b  = [5, 6, 7]
            #
            #>>>a.extend(b)
            #
            #[1, 2, 4, 5, 6, 7]
            #
            #>>>a = [1, 2, 4]
            #
            #>>>a.append(b)
            #
            #[1, 2, 4, [5, 6, 7]]
            
            # 创建抽取该特征以后的dataset
            retDataSet.append(reducedFeatVec)
    return retDataSet



##################选择最好的数据集划分方式#############################################
##################该函数实现选取特征，划分数据集，计算得出最好的划分数据集的特征##############
def chooseBestFeatureToSplit(dataSet):
    # 取出list中的第一个元素 再取长度-1 就为特征的个数
    # 比如a=[[1,1,'Yes'],[1,0,'No']] 那么a[0] = [1,1,'Yes'],然后len(a[0])=3,然后3-1=2，就是有两个特征
    numFeatures = len(dataSet[0]) - 1
    
    # 调用函数计算熵 Entropy(S),计算数据集中的原始香农熵
    baseEntropy = calcShannonEnt(dataSet)
    
    
    bestInfoGain = 0.0;bestFeature = -1
    # 为属性的索引值。由于从0开始。所以初始值设为-1
    
    
    for i in range(numFeatures):
        # 返回 dataset所有元素 中的 第1元素 并且为list
        featList = [example[i] for example in dataSet]
        
        # 在这里作用相当于matlab的 unique（） 去除重复元素
        # python中的集合(set)数据类型，与列表类型相似，唯一不同的是 set类型中元素不可重复
        uniqueVals = set(featList)
        
        newEntropy = 0.0
        for value in uniqueVals:
            
            # 调用函数返回属性i下值为value的子集
            subDataSet = splitDataSet(dataSet, i, value)
            
            #计算每个类别的熵
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        
        # 求信息增益
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    # 返回分类能力最好的属性索引值
    return bestFeature



#################多数表决的方法决定该叶子节点的分类####################################

def majorityCnt(classList):
    
    #声明 classCount是一个字典
    classCount = {}
    
    # 遍历字典的每一个元素
    for vote in classList:
        
        #如果这个元素（其实就是特征）不在字典的键里面，那么就是0，如果在，那么就加1
        if vote not in classCount.keys() : classCount[vote] = 0
        classCount[vote] += 1
    
    #classCount.items()将classCount字典分解为元组列表，operator.itemgetter(1)按照第二个元素的次序对元组进行排序，reverse=True是降序
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1), reverse=True)
    
    #找到次数最多的那个特征，把特征返回来
    return sortedClassCount[0][0]


################程序清单3-4 创建树的函数代码，给节点做标注#################################################

# #输入参数：数据集和标签列表
def createTree(dataSet,labels):
    
    # 取dataSet每个实例的最后一个元素，也即label,包含了所有类标签
    classList = [example[-1] for example in dataSet]
    
    # 类别完全相同则停止划分，取第一个就行了，第一个的个数等于所有的个数
    # 当计算在最后一列数据中与第一个值相同的元素个数与最后一列数据个数相同时，直接返回第一个元素值，意思是所有类标签都相同
    if classList.count(classList[0]) == len(classList):
        return classList[0] #当所有类都相等时停止分裂
    
    if len(dataSet[0]) == 1: #停止分裂时，没有更多的特征在数据集
        return majorityCnt(classList)
    
    # 返回最佳特征值划分的索引
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 得到最佳特征值索引的标签
    bestFeatLabel = labels[bestFeat]
    # 使用字典类型存储树的信息
    myTree = {bestFeatLabel:{}}
    
    # 从标签列表中删除最好特征值对应的那个标签
    del(labels[bestFeat])
    
    # 得到最佳特征值对应的数据集中的那一列数据组成列表
    featValues = [example[bestFeat] for example in dataSet]
    
    # 唯一化
    uniqueVals = set(featValues)
    
    # 遍历唯一化列表
    for value in uniqueVals:
        
        # 复制类标签，当函数参数是列表类型时，参数是按照引用方式传递的，保证每次调用函数时都不改变原始列表的内容，就是开一块新内存。
        subLabels = labels[:]
        
        # 等号前第一个中括号是指字典键值，键值可任意类型;第二个中括号是第一个键值延伸的嵌套的字典类型键值;在等号后，先把原数据集按特征值分开，然后递归调用该函数
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    #返回最终的字典信息
    return myTree



################### 程序清单 3-8 使用决策树的分类函数 ##################################
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


############# 程序清单 3-9  使用pickle模块存储决策树 ######################################
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close

def grabTree(filename) :
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)































