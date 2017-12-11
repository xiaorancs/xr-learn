# _*_coding:utf-8_*_
# Author: xiaoran
# Time: 2017-12-05 21:10
# perceptron

import numpy as np
import pandas as pd


class Perceptron(object):
    """Perception

    Parameters:
    -----------
    penalty:None, 'l2' ,'l1'
    alpha: float
        默认 0.0001
        如果使用正则化,就将所有的数据乘以alpha,
    max_iter: 可选
        默认 500
        最大的迭代次数 
    tol: float or None, optional
        默认 0.001,
        当 loss > pervious_loss - tol 停止.
    verbose: integer, optional
        默认 0
        显示训练过程的信息.
    shuffle: bool, optional 
        默认是 true    
        每一个迭代的时候打乱训练集.
    eta: double
        默认 1
        学习率
    ----------
    References:
        李航 <统计机器学习>
        林轩甜 <Perceptron PPT>
    """
    def __init__(self, penalty=None, alpha=0.0001, max_iter=500, 
                 tol=None, verbose=0, shuffle=True, eta=1):
        self.__penalty = penalty
        self.__alpha = alpha
        self.__max_iter = max_iter
        self.__tol = tol
        self.__verbose = verbose
        self.__shuffle = shuffle
        self.__eta = eta
        self.__W = None
        self.__b = None


    def __str__(self):
        '''
        格式化输出
        '''
        return "%s" % (self.__penalty)

    def __resample(self,X,start,end):
        '''X[start,end]
            交换
        '''
        while start < end:
            temp = X[start]
            X[start] = X[end]
            X[end] = temp
            start += 1
            end -= 1
        return X

    def __shuffleData(self,X,y):
        '''随机打乱数据
            随机数index,顺时针旋转index次
            使用算法,三次翻转法
        '''
        index = np.random.randint(len(X))
        # X
        self.__resample(X,0,index)
        self.__resample(X,index+1,len(X)-1)
        self.__resample(X,0,len(X)-1)
        
        # y
        self.__resample(y,0,index)
        self.__resample(y,index+1,len(X)-1)
        self.__resample(y,0,len(X)-1)
        
        return X,y


    def sign(self,X):
        '''
        符号函数,
        x >= 0: return 1
        else return 0 
        '''
        return 1 if x>=0 else 0


    def fit(self,X,y):
        '''
        X: 所有的训练数据,多维数据或者ndarray
        y: {1,0},如果默认是是0,
        '''
        # 检查数据
        try:
            X = np.array(X)
        except:
            print("训练数据不满足要求")
        X = X * self.__alpha       

        label = set(y)
        if len(label) > 2:
            print("label标签不满足二分类的要求,")
            exit(0)

        # print(X.shape)
        # 开始迭代训练
        iter = 0
        pervious_loss = None
        cur_loss = None
        self.__W = np.zeros(X.shape[1])
        self.__b = 0
        nums = X.shape[0]
        while iter < self.__max_iter:
            temp_loss = 0
            for i in range(nums):
                target = -1 if y[i] == 0 else 1
                # 更新权重,误分类的点
                if target * (np.sum(X[i] * self.__W) + self.__b) <= 0:
                    temp_loss -= target * (np.sum(X[i] * self.__W) + self.__b)
                    self.__W = self.__W + self.__eta * target * X[i]
                    self.__b = self.__b + self.__eta * target
                
            iter += 1
            # X,y = self.__shuffleData(X,y)
            # print(X,y)
            pervious_loss = cur_loss
            cur_loss = temp_loss
            
            if self.__verbose != 0 and iter % self.__verbose == 0:
                print('第%d的损失值是------------->%f' % (iter,cur_loss))

            # 设置了tol参数,而且满足条件,进直接返回
            if pervious_loss != None and self.__tol != None  and np.abs(pervious_loss - cur_loss) < self.__tol:
                return
            
        # 返回所有的参数self, penalty=None, 0.0001, 500, 
        return self                 
        # print("Perceptron(penalty=%s, alpha=%f, max_iter=%d, tol=%f, verbose=%f, shuffle=%b, eta=%f)" 
        #      % (self.__penalty, self.__alpha, self.__max_iter, self.__tol, self.__verbose, self.__shuffle, self.__eta))

    
    def get_params(self):
        '''
        return 参数W和b
        '''
        return self.__W / self.__alpha, self.__b / self.__alpha

    def source(self,X,y):
        """
        参数:
            X:形如训练数据中的X
            y:对应训练数据的y,和X的对应标签
            return 正确率
        """        
        # 检查数据
        try:
            X = np.array(X)
        except:
            print("训练数据不满足要求")
        
        X = X * self.__alpha
        label = set(y)
        if len(label) > 2:
            print("label标签不满足二分类的要求,")
            exit(0)

        y = np.array(y)
        y_pre = np.sign(np.sum(X * self.__W +self.__b,axis=1))
        y_pre[y_pre==-1]=0
        acc = np.sum(y_pre == y) / len(y)

        print("验证集上的得到acc = ",acc)
        return acc
    
    def predict(self,X):
        '''
        X:形如训练数据,
        return y_pre 
        '''
        AXIS = 1
        y_pre = None
        try:
            X = np.array(X)
        except:
            print("预测数据出现了异常.")
        # 只有一个元素
        if len(X.shape) == 1:
            X = np.array([X])
        
        X = X * self.__alpha
        y_pre = np.sign(np.sum(X * self.__W + self.__b,axis=AXIS))
        y_pre[y_pre==-1]=0

        return y_pre
