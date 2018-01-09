# _*_ encoding : utf-8 _*_
# Author: xiaoran
# Time: 2017-12-30 17:53
# Email: xiaoranone@gmail.com

import numpy as np
import pandas as pd
import scipy as sp

class kNN(object):
    '''kNN(k-NearestNeighbor)
    
    参数：
        k_neighbors: int defaute 3
            使用的数据的个数。
        metric: string default "l2"
            距离计算函数，可以选择['cosine','l1','l2'],
            可以自行编写其他距离函数

    fit(X,y)
    get_params()
    source(X,y)
    predict(x)
    '''
    def __init__(self,  k_neighbors = 3, metric = 'l2'):
        '''初始化

        '''
        self.__k_neighbors = k_neighbors
        self.__metric = metric

        self.X = None
        self.y = None

    def fit(self, X, y, check_input=True):
        '''fit
        参数：
            X: 多维数组(narray), 或者 DataFrame, [n_example x n_features]
            y: [n_example, 1]
        '''
        if check_input:
            if len(X) != len(y):
                raise ValueError("X和y的长度不一致，请检查输入！")

        self.X = np.array(X)
        self.y = np.array(y)

        return self
    

    def __l1(self, a, b):
        if len(a) != len(b):
            raise ValueError("长度不一致，请检查输入！")
        a = np.array(a)
        b = np.array(b)
        return np.sum(np.abs(a - b))
    
    def __l2(self, a, b):
        if len(a) != len(b):
            raise ValueError("长度不一致，请检查输入！")
        a = np.array(a)
        b = np.array(b)
        return np.sum((a - b)**2)
    
    def __cosine(self, a, b):
        if len(a) != len(b):
            raise ValueError("长度不一致，请检查输入！")
        a = np.array(a)
        b = np.array(b)
        return np.sum(a*b) / (np.sum(a**2) * np.sum(b**2)) 
    

    def __predictOne(self, x, check_input = True):

        if check_input == True:
            if len(x) != self.X.shape[1]:
                raise ValueError("数据x和原始数据特征个数不相同！")
        
        rows,clomns = self.X.shape

        if self.__metric == 'l2':
            dist = self.__l2
        elif self.__metric == 'l1':
            dist = self.__l1
        else:
            dist = self.__cosine

        distlist = []
        for i in range(rows):
            distlist.append(dist(self.X[i], x))
        
        distlist = np.array(distlist)

        # print(distlist)
        # 前k个最小值对应的下标
        sortMinK = distlist.argsort()[:self.__k_neighbors]

        # print(sortMinK)

        # 最小距离对应的lable
        labelK = self.y[sortMinK]
        
        # print(labelK)
        most_label = np.argmax(np.bincount(labelK))

        return most_label
    
    def predict(self,X,check_input=True):

        X = np.array(X)
        if check_input:
            if len(X.shape) == 1:
                X = np.array([X])
        y_ = []

        for i in range(X.shape[0]):
            y_.append(self.__predictOne(X[i]))

        return np.array(y_)
    
    def scoure(self, X, y, check_input = True):
        if check_input == True:
            if len(x) != len(y):
                raise ValueError("X和y的长度不一致，请检查输入！")

        y_ = self.predict(X)

        y = np.array(y)

        aurr = np.sum(y_ == y)

        print("验证的正确集是: ", 1.0 * aurr / len(y) )

        return aurr
    