# _*_coding:utf-8_*_
# Author: xiaoran
# Time: 2017-12-08 21:10
# DecisionTreeClassifier

import numpy as np
import scipy as sp
import pandas as pd


class DecisionTreeClassifier(object):
    """决策树分类器,主要基于ID3和C4.5
    criterion: string optional (default="gini")
        选择特征的基础:
        entropy [enrtopy]: 熵  for ID3
        information_gain [i_gain]: 信息增益 for ID3
        information_gain_ratio [i_gain_r]: 信息增益比 for C4.5 
        gini [gini]: gini 指数 for CART
    max_depth: int or None, optional (default = None)
        最大深度,if None, 则知道所有叶子都是一个类,或者剩下min_sample_split个例子.(用来防止过拟合)
    min_sample_split: int float, optional (default=2)
        剩余最少几个例子的时候不在进行分割,使用例子中出现最多的类,作为这个叶子节点的标签label.(用来防止过拟合)
        IF float向上取整.
    属性:
    classes_ : 所有的类别.
    feature_importances_: 
        特征重要性,根据分类的正确数进行递减排序,一共两维数据,[(feature1, nums1),...,(featureN,numsN)],首先给出一维
        这是根据创建树的过程生成的,并不对应验证集的结果,而且返回的对应特征的列编号.
        numpy: [column0,...,columni,...]
        DataFrame: [DataFrame.columns]
    tree_: 决策树的原型.
    
    实现函数:
    fit(),predict(),apply(), score(),

    """
    def __init__(self,criterion='i_gain_r',max_depth=None,min_sample_split=2):
        '''构造函数

        '''
        self.__criterion = criterion
        self.__max_depth = max_depth
        self.__min_sample_plite = min_sample_split
        self.__featureLen = None
        self.__tree_ = None
        self.classes_ = None
        self.feature_importances_ = []
        self.tree_ = None
    
    def __check_array(self,x):
        '''
        检查x的数据,
        None:自动填充0,
        if isinstance(x,list)--> x = np.array(x) 
        if x只有一个元素,将其变为二维的数据.x = np.array([x])
        '''
        if isinstance(x,list):
            x = np.array(x)
        
        if self.__featureLen == None:
            self.__featureLen = x.shape[1]

        if len(x.shape) == 1:
            x = np.array([x])
        
        if x.shape[1] != self.__featureLen:
            raise ValueError("输入数据的格式与训练数据的格式不匹配.")

        return x

    def __spliteDataWithFeature(self,data,featureColumn,dataType='numpy'):
        '''根据给定的特征,分割数据集,返回对应的数据子集,这里的特征使用列号[0,...,n]给出, 
        参数:
        data: 被分割的数据
        featureColumn: 特征的列索引
        dataType: 数据类型,默认"ndarray",还有一种是pd.DataFrame,注意numpy的ndarry兼容DataFrame

        return 对应的分离数据和对应得到这个子集的特征的值
        '''
        splitdataSet = []

        if dataType == 'numpy':
            featureSet = set(data[:,featureColumn])
            # print("featureSet",featureSet)
            for feature in featureSet:
                tmp = np.copy(data[data[:,featureColumn] == feature])
                tmp = np.delete(tmp,featureColumn,axis=1)                
                splitdataSet.append(tmp)

        else : # dataType == 'DataFrame'
            columns = data.columns
            featureSet = set(data[columns[featureColumn]])
            for feature in featureSet:
                tmp = data[data[columns[featureColumn]] == feature].drop(columns[featureColumn],axis=1)
                splitdataSet.append(tmp)

        return splitdataSet,list(featureSet)

    def __calculateEntropy(self,labelSet):
        '''计算信息熵,
        参数:
            labelSet:数据对应的类的集合
        return 对应的熵
        '''
        # 重复确认数据类型
        labelSet = np.array(labelSet)
        # print("labelSet")
        # print(labelSet)
        # 总长度
        length = len(labelSet)

        entropy = 0
        classes = set(labelSet)
        for c in classes:
            p = 1.0 * np.sum(labelSet == c) / length
            entropy -= p * np.log2(p)
        return entropy

    def __calculateGini(self,labelSet):
        '''计算信息gini指数,
        参数:
            labelSet:数据对应的类的集合
        return 对应的给你指数
        '''
        # 重复确认数据类型
        labelSet = np.array(labelSet)
        # 总长度
        length = len(labelSet)

        gini = 1
        classes = set(labelSet)
        for c in classes:
            gini -= (np.sum(labelSet == c) / length) ** 2

        return gini

    def __getBestFeature(self,data):
        '''根据指定的方式计算给定数据集的最有特征,
        参数:
        data:给定的数据集
        criterion: 计算的方式, 默认是gini指数,[entropy, i_gain, i_gain_r, gini]
        return 返回的是特征的列编号,从0开始
        根据每一列的值进行计算,得到最好的特征对应的列
        注意gini指数对应的CART,使用的是二叉树的形式.
        '''
        data = np.array(data)
        # print("bestfeature=",data)
        if self.__criterion == 'gini':
            origin_gini = self.__calculateGini(data[:,-1])
            pass
        else:
            # 计算原始的熵
            origin_entropy = self.__calculateEntropy(data[:,-1])

            # print(origin_entropy)
            # 计算每一列特征,对应的熵,以列号作为标识
            row, column = data.shape
            column -= 1
            
            # 纪录不同特征分割后的信息熵
            entrop_split = [0 for i in range(column)]

            for i in range(column):
                splitdataSet = self.__spliteDataWithFeature(data,i)[0]
                # print(i,"------------------------>")
                # print(splitdataSet)
                for subSet in splitdataSet:
                    # print(subSet.shape)
                    entrop_split[i] += (subSet.shape[0] * 1.0 / row) * self.__calculateEntropy(subSet[:,-1])

            entrop_split = np.array(entrop_split)
            # 信息熵的增益 = 原始熵 - 特征选择的信息熵
            entrop_split_diff = origin_entropy - entrop_split

            # 信息增益比
            entrop_split_diff_ratio = entrop_split_diff / entrop_split

            # 使用的评测标准是信息熵或信息增益,都是用信息增益,对应ID3,最大化信息增益
            if self.__criterion in ['entropy','i_gain'] :
                bestFeature = np.argmax(entrop_split_diff)
            # 信息增益比,对应C4.5
            if self.__criterion == 'i_gain_r':
                bestFeature = np.argmax(entrop_split_diff_ratio)

        # print(entrop_split)
        # print(entrop_split_diff)            
        # print(entrop_split_diff_ratio)

        return bestFeature           

    def __createDecisionTree(self,data,depth,columnindex):
        '''决策树的分类算法,主要就是创建决策树,递归创建决策树.
        参数: data:包含最后一列的label
        return 字典类型的决策树,self.tree_
        '''
        # 数据归一化为np.ndarray
        data = np.array(data)
        # 根据算法和参数设置递归出口
        labels = set(data[:,-1])

        # 只剩下唯一的类别时,停止,返回对应类别
        if len(labels) == 1:
            return list(labels)[0]
        # 遍历完所有特征时,只剩下label标签,就返回出现字数最多的类标签
        if data.shape[1] == 1:
            return np.argmax(np.bincount(data[:,-1]))

        if self.__max_depth != None and depth > self.__max_depth:
        # 如果剩余的样本数小于等于给定的参数 min_sample_plite,则返回类别中最多的类
            return np.argmax(np.bincount(data[:,-1]))

        # 根据参数返回值,树的深度大于给定的参数max_depth,则返回类别中最多的类
        if self.__min_sample_plite >= data.shape[0]:
            return np.argmax(np.bincount(data[:,-1]))

        bestFeature = self.__getBestFeature(data)
        bestFeatureLabel = columnindex[bestFeature]
        # 纪录删除的类别号,即所有最有的列
        # 根据创建树的特征,创建决策树的过程中以此纪录特征
        self.feature_importances_.append(columnindex[bestFeature])

        del(columnindex[bestFeature])
        # print(bestFeature)

        decisionTree = {bestFeatureLabel:{}}

        spliteDataSub, featureSetValue = self.__spliteDataWithFeature(data,bestFeature)

        # print(spliteDataSub)
        # print(featureSetValue)
        for i in range(len(featureSetValue)):
            subcolumnindex = columnindex
            decisionTree[bestFeatureLabel][featureSetValue[i]] = self.__createDecisionTree(spliteDataSub[i],depth+1,columnindex)

        return decisionTree

    def fit(self,X,Y,check_input=True):
        '''params:
        X: 多维数组,numpy.narray, DataFrame, n X m,m个特征
        Y: n X 1, 与X长度对应
        '''
        if len(X) != len(Y):
            raise ValueError("特征集和label的长度不匹配")

        if isinstance(X,list) and check_input == True:
            X = self.__check_array(X)
            Y = np.array(Y)
        
        # 设置类别参数,得到所有的类
        self.classes_ = list(set(Y))
        # 合并数据和label便于后期出来,最后一列是label
        # 多维数组的类型
        if isinstance(X,np.ndarray):
            data = np.c_[X,Y]
            # 得到类的标号
            columnindex = ["column"+str(i) for i in range(X.shape[1])]
            columnindexInner = [i for i in range(X.shape[1])]
            
        # pandasDataFrame类型
        if isinstance(X,pd.DataFrame):
            data = pd.concat([X,Y],axis=1)        
            # 得到类的标号
            columnindex = list(X.columns)
            columnindexInner = [i for i in range(len(columnindex))]
            
        self.__featureLen = len(columnindex)
        
        self.tree_ = self.__createDecisionTree(data,0,columnindex)
        # 设置内部索引树便于predict,但是浪费了一倍的时间
        self.__tree_ = self.__createDecisionTree(data,0,columnindexInner)
        

        return self.tree_

    def __predictUtil(self,tmpTree,x):
        """预测单一元素的类别,
        x:一个数据
        这里使用递归调用,因为字典的格式是:(index,dict(index,dict))的格式,可以使用递归
        return lable
        """
        label = self.classes_[0] # 防止有没有出现的值,直接返回任意一个类
    
        if type(tmpTree) == dict:
            firstIndex = list(tmpTree.keys())[0]
            secondDict = tmpTree[firstIndex]
            
            for key in secondDict.keys():
                if x[firstIndex] == key: # 对应的值等于索引的值
                    if type(secondDict[key]) == dict:
                        label = self.__predictUtil(secondDict[key],x)
                    else:
                        label = secondDict[key]
        else:
            label = tmpTree
        
        return label

    def predict(self,x,check_input=True):
        '''
        预测分类,x,形如训练数据的格式,
        return lable
        '''
        if check_input == True:
            x = self.__check_array(x)
        
        pre_y = []
        tmpTree = self.__tree_
        x = np.array(x)
        for i in range(x.shape[0]):
            pre_y.append(self.__predictUtil(tmpTree,x[i]))
        
        return np.array(pre_y)
    
    def score(self,x,y):
        '''
        x,y : 形如训练数据
        评测正确率
        '''
        if len(x) != len(y):
            raise ValueError("特征集和label的长度不匹配")

        pre_y = self.predict(x)
        y = np.array(y)

        scoreValue = np.sum(pre_y == y) * 10 / len(y)
        print("模型在该验证集下的正确率=",scoreValue)
        return scoreValue
