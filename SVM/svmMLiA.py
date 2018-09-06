
####################### 程序清单 6-1 SMO算法中的辅助函数 ################
import random
from  numpy import * 
def loadDataSet(filename):
    dataMat=[];labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m): #i表示alpha的下标，m表示alpha的总数
    j=i
    while(j==i):
        j=int(random.uniform(0,m)) #简化版SMO，alpha随机选择
    return j

def clipAlpha(aj,H,L): #辅助函数，用于调整alpha范围
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj

################# 程序清单 6-2 简化版SMO算法 ###################
#SMO simple algorithm
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):#toler表示容错率 常数C
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    b = 0; m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0   #  标记alpha是否被优化
        for i in range(m):
            # fXi是预测的类别
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            # Ei表示误差
            Ei = fXi - float(labelMat[i])# 预测结果和真实结果比对，计算误差
            # 对alpha进行优化，同时检查alpha的值满足两个条件：if
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)# 随机选择第二个alpha
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy(); # 分配内存 稍后比较误差
                if (labelMat[i] != labelMat[j]): # 计算L H用于将alpha[j]调整到0—C之间
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print ("L==H"); continue  # eta为alpha[j]的最优修改量
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print ("eta>=0"); continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                # 检查alpha[j]是否有轻微改变
                if (abs(alphas[j] - alphaJold) < 0.00001): print ("j not moving enough"); continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
                                                                        #the update is in the oppostie direction
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print ("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print ("iteration number: %d" % iter)
    return b,alphas

################## 程序清单6-3 完成版Platte SOM的支持函数 ############
class optStruct:  #保存所有重要值，实现对成员变量的填充
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters
        import numpy as np 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = np.mat(zeros((self.m,2))) #误差缓存
        #self.K = mat(zeros((self.m,self.m)))
        #for i in range(self.m):
         #   self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

def calcEk(oS,k):#计算误差并返回
    import numpy as np
    fXk=float(np.multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T))+oS.b
    Ek=fXk-float(oS.labelMat[k])
    return Ek

def selectJ(i,oS,Ei): #选择第二个alpha值以保证每次优化的最大步长（内循环）change English~
    import numpy as np
    maxK=-1; maxDeltaE=0; Ej=0
    oS.eCache[i]=[1,Ei] #input Ei
    validEcacheList=np.nonzero(oS.eCache[:,0].A)[0] #set Ei valid in cache，nonzero is a list
    #choose the num which change most, if the first choice: use random
    if(len(validEcacheList))>1:
        for k in validEcacheList:
            if k==i:continue
            Ek=calcEk(oS,k)
            deltaE=abs(Ei-Ek)
            if(deltaE>maxDeltaE):
                maxK=k; maxDeltaE=deltaE; Ej=Ek
        return maxK,Ej
    else:
        j=selectJrand(i,oS.m)
        Ej=calcEk(oS,j)
    return j,Ej

def updateEk(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

################### 程序清单 6-4 完整Platt SMO 算法中的优化例程 #########
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print ("L==H"); return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
        if eta >= 0: print ("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print ("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

################ 程序清单 6-5 Platt SMO 的外循环代码 ###############
def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T   #linear kernel
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
    else: raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K

class optStruct:  #保存所有重要值，实现对成员变量的填充
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters
        import numpy as np
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = np.mat(zeros((self.m,2))) #误差缓存
        self.K = np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

def calcEk(oS, k):#计算误差
    import numpy as np
    fXk = float(np.multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i,oS,Ei): #选择第二个alpha值以保证每次优化的最大步长（内循环）change English~
    import numpy as np
    maxK=-1; maxDeltaE=0; Ej=0
    oS.eCache[i]=[1,Ei] #input Ei
    validEcacheList=np.nonzero(oS.eCache[:,0].A)[0] #set Ei valid in cache，nonzero is a list
    #choose the num which change most, if the first choice: use random
    if(len(validEcacheList))>1:
        for k in validEcacheList:
            if k==i:continue
            Ek=calcEk(oS,k)
            deltaE=abs(Ei-Ek)
            if(deltaE>maxDeltaE):
                maxK=k; maxDeltaE=deltaE; Ej=Ek
        return maxK,Ej
    else:
        j=selectJrand(i,oS.m)
        Ej=calcEk(oS,j)
    return j,Ej

def updateEk(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print ("L==H"); return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
        if eta >= 0: print ("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print ("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

#full Platt SMO
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    import numpy as np
    oS = optStruct(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    # 当迭代超过maxIter或者整个数据集都未对任意alpha进行修改时，退出循环
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)#use innerL to choose the second alpha
                print ("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print ("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True
        print ("iteration number: %d" % iter)
    return oS.b,oS.alphas

#由alpha得到权值w

def calcWs(alphas,dataArr,classLabels):
    import numpy as np
    X = np.mat(dataArr); labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(X)
    w = np.zeros((n,1))
    for i in range(m):
        w += np.multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

##############  计算超平面及w ##############################
def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i], X[i,:].T)
    return w

############ 程序清单 6-6 核转换函数 #######################
def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T   #linear kernel
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
    else: raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K
 
class optStruct: #建立一个数据结构来保存所有的重要值
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag（是否有效的标志位），第二列是实际的E值（误差值）
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

########## 程序清单6-7 使用核函数时需要对innerL()和calcEk()函数的修改     
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek
        
def selectJ(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    #选择第二个alhpa值以保证在每次优化中采用最大步长
    oS.eCache[i] = [1,Ei]  #set valid（将即将使用的Ei值设为有效） #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1: #选择其中使改变最大的那个E值
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej
 
def updateEk(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]
'''
功能：和smoSimple函数一样（包括选择第二个乘子）
'''
def innerL(i, oS):
    """
    优化的SMO算法
    Parameters：
        i - 标号为i的数据的索引值
        oS - 数据结构
    Returns:
        1 - 有任意一对alpha值发生变化
        0 - 没有任意一对alpha值发生变化或变化太小
    """
    #步骤1：计算误差Ei
    Ei = calcEk(oS, i)
    #优化alpha,设定一定的容错率。
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        #使用内循环启发方式2选择alpha_j,并计算Ej
        j,Ej = selectJ(i, oS, Ei)
        #保存更新前的aplpha值，使用深拷贝
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        #步骤2：计算上下界L和H
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        #步骤3：计算eta
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
        if eta >= 0:
            print("eta>=0")
            return 0
        #步骤4：更新alpha_j
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej)/eta
        #步骤5：修剪alpha_j
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        #更新Ej至误差缓存
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("alpha_j变化太小")
            return 0
        #步骤6：更新alpha_i
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        #更新Ei至误差缓存
        updateEk(oS, i)
        #步骤7：更新b_1和b_2
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        #步骤8：根据b_1和b_2更新b
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0
'''
功能：外循环（选择第一个乘子）
'''
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    import numpy as np
    oS = optStruct(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler, kTup) #建立一个数据结构来保存所有的重要值
    iter = 0
    entireSet = True; alphaPairsChanged = 0 #alpha改变标志位
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0 
        if entireSet:   #go over all 遍历整个数据集
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
                print ("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print ("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True  
        print ("iteration number: %d" % iter)
    return oS.b,oS.alphas

########## 程序清单6-8 利用核函数进行分类的径向基测试函数 ##########3
'''
功能：利用核函数进行分类
'''
def testRbf(k1=0.1): 
    import numpy as np
    dataArr,labelArr = loadDataSet('/Users/bindo/Desktop/Machine Learning/machinelearninginaction/Ch06/testSetRBF.txt')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    datMat=np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    svInd=np.nonzero(alphas.A>0)[0] #找到非零的alphas值，从而得到了所需要的支持向量
    sVs=datMat[svInd] #get matrix of only support vectors
    labelSV = labelMat[svInd];#得到了支持向量的类别标签，
    print ("there are %d Support Vectors" % shape(sVs)[0])
    m,n = np.shape(datMat)
    predict_label0_index = []
    predict_label1_index = []
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1)) #只利用支持向量就可进行分类
        predict = array(kernelEval.T * multiply(labelSV,alphas[svInd]) + b) #计算预测值
        if sign(predict) == -1: #如果预测的标签是-1，则保存一个列表中
            predict_label0_index.append(i)
        elif sign(predict) == 1: #如果预测的标签是-1，则保存另一个列表中
            predict_label1_index.append(i)
        if sign(predict)!=sign(labelArr[i]): errorCount += 1 #利用sign函数，判断预测是否正确
    print ("the training error rate is: %f" % (float(errorCount)/m))
    plotfig_kernel(dataArr,predict_label0_index,predict_label1_index,alphas)#画图
    dataArr,labelArr = loadDataSet('testSetRBF2.txt') #再用另一个数据集进行测试，后面代码和前面一样
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print ("the test error rate is: %f" % (float(errorCount)/m))


############ 画利用核函数进行分类的图 ###########
'''
功能：画利用核函数进行分类的图
'''
def plotfig_kernel(xMat,predict_label0_index,predict_label1_index,alphas):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in predict_label0_index: #预测标签为-1的标为红色三角形
        ax.plot(xMat[i][0],xMat[i][1],'r^')
    for i in predict_label1_index: #预测标签为1的标为蓝色正方形
        ax.plot(xMat[i][0],xMat[i][1],'bs')
    plt.show()

############### 手写系统把数据处理成分类器需要格式的函数 ##########################
def img2vector(filename):
    import numpy
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

############### 程序清单 6-9 基于SVM的手写数字识别 ###############3
def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else : hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s'%(dirName, fileNameStr))
    return trainingMat, hwLabels
def testDigits(kTup = ('rbf',10)):
    dataArr,labelArr = loadImages('/Users/bindo/Desktop/Machine Learning/machinelearninginaction/Ch06/digits/trainingDigits')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000,kTup)
    dataMat = mat(dataArr);labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A>0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd];
    print("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i,:],kTup)
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):errorCount += 1
    print("the training error rate is: %f" %(float(errorCount)/m))
    dataArr, labelArr = loadImages('/Users/bindo/Desktop/Machine Learning/machinelearninginaction/Ch06/digits/testDigits')
    errorCount = 0
    dataMat = mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],kTup)
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):errorCount += 1
    print("the test error rate is: %f" % (float(errorCount)/m))
    

