# recommend system(推荐系统)
    推荐系统现在应用很普遍,应该说任何网站和应用都有推荐系统的影子.目的是给用户推荐你喜欢或者将要喜欢的物品后者朋友;个人认为网易音乐的推荐做的不错.
## 协同过滤
    协同过滤算法是最简单,应用最多的推荐算法.主要有两种:基于用户的协同过滤算法和基于物品的协同过滤算法.
    基于用户的协同过滤算法:
        对于用户A的推荐,我们使用与A相近"臭味相投"的用户[B,C,D...],将这些用户的喜欢的物品推荐给A.
        当然我们和可以添加权重P_AB表示A和B相似的程度.
    基于物品的协同过滤算法:
        对于用户A的推荐,假设A买过物品集合是[I_a,I_b,...,],推荐给A与其购买的物品集相似的物品.我们认为你如果买了羽毛球排,那么
        你一定也会想要买羽毛球.
    协同过滤算法的实现:
    fit(train-uers,train-items),
    score(vali-usres,vali-items)
    predict(uers)
    '''
    协同过滤算法的实现,
    主要的实现功能函数:
        fit(users,items),
        score(users,items),
        predict(users)
    
    类的参数:
    criterion: 默认 'user',可以选择['user','item']
        协同过滤的算法的准则,是基于用户或者基于物品,默认是基于user的协同过滤
    similarity: 计算相似性, 默认 'cosine',使用余弦相似度
        使用的相似度计算函数['cosine','jaccard','pearson'],分别是余弦和皮尔逊相关系数
    N: 默认是 5 , int 
        使用N个相似的人或者物品进行推荐,默认是5
    K: 默认 5, int
        给用户推荐几个物品, 整数的性质.如果够5个问题就使用最热门物品填充.
    contain: 默认是True
        是否推荐用户过去买过的物品,contain=True,默认进行推荐
        false: 不推荐过去买过的物品
    注意: N和K会直接影响评测标注.

    类的属性值:
    similarDict: 相似性矩阵
    popularItemSet: 流行的物品集合,默认纪录前100个.
    '''
    + 使用样例:


    ```
    import CF
    data = {'u1':{'a':1,'b':2,'c':3},'u2':{'a':1,'d':2},        'u3':{'a':2,'c':1,'d':5,'e':1},'u4':{'f':2,'e':1}}
    # 默认基于用户的协同过滤
    cf = CF.CF()
    cf.fit(data)
    cf.predict(['u1'])

    # 基于物品的协同过滤
    cf = CF.CF(criterion='item')
    cf.fit(data)
    cf.predict(['u1'])

    # 相似度矩阵
    cf.similarityMatrix

    cf.source(['u1','u2'],[['a','b','f'],['a','d','g']])
    ```
    