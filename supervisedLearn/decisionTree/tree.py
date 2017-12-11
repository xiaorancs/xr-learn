# _*_coding:utf-8_*_
# Author: xiaoran
# Time: 2017-12-08 21:10
# DecisionTreeClassifier

import numpy as np
import scipy as sp


class DecisionTreeClassifier(object):
    """决策树分类器,主要基于ID3和C4.5
    criterion: string optional (default="gini")
        选择特征的基础:
        entropy: 熵  for ID3
        information gain: 信息增益 for ID3
        information gain ratio: 信息增益比 for C4.5 
        gini impurity: gin 指数 for CART
    max_depth: int or None, optional (default = None)
        最大深度,if None, 则知道所有叶子都是一个类,或者剩下min_sample_split个例子.(用来防止过拟合)
    min_sample_split: int float, optional (default=2)
        剩余最少几个例子的时候不在进行分割,使用例子中出现最多的类,作为这个叶子节点的标签label.(用来防止过拟合)
        IF float向上取整.
    属性:
    classes_ : 所有的类别.
    feature_importances_: 
        特征重要性,根据分类的正确数进行递减排序,一共两维数据,[(feature1, nums1),...,(featureN,numsN)]
    
    """

