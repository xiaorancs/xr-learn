# _*_coding:utf-8_*_
# Author: xiaoran
# Time: 2017-12-19 19:43
# LogisticRegerssion

import numpy as np
import numbers

class LogisticRegerssion(object):
    '''LogisticRegression算法实现
    参数:
        penalty: str, 'l1' or 'l2',默认: 'l2'
            使用的正则化的方式,用来防止过拟合.
        tol: float, 默认: 1e-4
            停止的容忍误差
        C: float, 默认: 1.0
            逆正则化,平衡误差和正则项的值
        random_state: int, 默认: None
            根据随机数种子,随机打乱数据.
        max_iter: int, 默认: 100
            最大的迭代次数,针对牛顿法等
        verbose: int, 默认: 0
            是否显示训练信息,每个verbose显示一次
    属性:
        coef_: array,决策函数的系数,
        intercept_: array, 决策函数的截
    '''
    def __init__(self,penalty='l2',tol=0.0001,C=1,random_state=None,max_iter=100,verbose=0):
        self.__penalty = penalty
        self.__tol = tol
        self.__C = C
        self.__random_state = random_state
        self.__max_iter = max_iter
        self.__verbose = verbose

        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None
    
    def __prenalty_function(self,x,y,w,b,penalty='l2'):
        '''计算一次数据的梯度,根据参数,使用正则化, 默认: 'l2'
        参数:
            x: 一个样本数据
            y: 样本x的对应标签
            w,b: 当前的参数 
        return W,b的梯度
        '''
        util = np.exp(-y*np.sum(x*w+b))

        if penalty == 'l1':
            w[w>0] = 1
            w[w<=0] = -1
        
        gd_w = w - (x * self.__C * util * y) / (util + 1)
        gd_b = - (self.__C * util * y) / (util + 1)
        # print(gd_w,gd_b)
        return gd_w, gd_b

    def sigmoid(self,x):
        '''计算sigmoid的值,
        return v
        '''
        return 1.0 / (1 + np.exp(-x))

    def __gradientDescent(self,X,y,w,b):
        '''使用梯度下降计算W和,b,计算一轮所有的数据
        参数:
            X: 数组或者矩阵, shape(n_samples,n_features)
            y: array, shape(n_samples,)
            w,b: 上一轮的参数,初始值是0.
        return grad_W,grad_b
        '''
        # w = np.zeros(X[0].shape)
        # b = 0
        tmpw = w
        if self.__penalty != 'l2':
            tmpw[tmpw>0] = 1
            tmpw[tmpw<=0] = -1
        
        util = np.exp(-y * (np.sum(X*w,axis=1)+b)) 
        n = X.shape[0]

        # print(util)
        # 平均梯度下降
        grad_w = tmpw - self.__C * np.dot(X.T , util * y) / np.sum(util+1) / n
        grad_b = np.sum(- self.__C * (util*y) / (util+1)) / n

        w -= grad_w
        b -= grad_b
        return w, b


    def fit(self,X,y,check_input=True):
        '''
        参数:
            X: array [n_sample, n_features]
            y: array [n_sample]
        return self
        '''
        if not isinstance(self.__C, numbers.Number) or self.__C < 0:
            raise ValueError("正则化的系数必须是正数; got (C=%r)" % self.__C)
        if not isinstance(self.__max_iter, numbers.Number) or self.__max_iter < 0:
            raise ValueError("最大迭代次数必须是正整数; got (max_iter=%r)" % self.__max_iter)
        if not isinstance(self.__tol, numbers.Number) or self.__tol < 0:
            raise ValueError("停止的阈值必须是正数; got (tol=%r)" % self.__tol)

        if check_input == True:
            X = np.array(X)
            y = np.array(y)
        # print(X)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        y[y==0]=-1
        if n_classes > 2: # 类别超过两个
            raise ValueError("类的种类超过两个类,请检查数据!")
        
        # 纪录上次的计算值

        # 初始化参数
        w = np.zeros(X[0].shape)
        b = 0

        pre_loss = np.sum((np.sum(X * w) + b))

        for iter in range(self.__max_iter):
            if self.__verbose > 0:
                print("第%d迭代次数-----> %f" % (iter, pre_loss))

            w,b = self.__gradientDescent(X,y,w,b)

            cur_loss = np.sum((np.sum(X * w) + b))
            if np.abs(cur_loss - pre_loss) < self.__tol:
                break
            pre_loss = cur_loss

        self.coef_ , self.intercept_ = w, b

        return self

    def predict_proba(self,X,check_input=True):
        '''预测函数,
        参数:
            X: 数组或者矩阵,和训练数据x格式一样

        return array,表示1的概率
        '''
        if check_input == True:
            X = np.array(X)
            if len(X.shape) == 1: # 只有一个数据
                X = np.array([X])
        
        v = np.exp(np.dot(self.coef_, X.T) + self.intercept_)

        prob_y = v / (1+v)

        return prob_y
    
    def predict(self,X):
        '''预测数据的种类
        X: 数组,列表; 形如训练数据的格式
        return 对应的类别,1 or 0
        '''
        y_ = self.predict_proba(X)
        y_[y_ >= 0.5] = 1
        y_[y_ < 0.5] = 0
        
        return y_

    def source(self,X,y):
        '''计算正确率,
        参数:
            X: 数组,和训练数据一致
            y: 对应的类
        '''
        y_ = self.predict(X)
        # print(y_)
        y = np.array(y)
        acc = np.sum(y_ == y) * 1.0 / len(y)
        print("正确率:",acc)
        return acc
        