# _*_coding:utf-8_*_
# Author: xiaoran
# Time: 2017-12-16 13:17
# collaboratuve filter 
import numpy as np
import scipy as sp

class CF(object):
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
    def __init__(self,criterion='user',similarity='consine',N=5,K=5,contain=False):
        self.__criterion = criterion
        self.__similarity = similarity
        self.__N = N
        self.__K = K
        self.__contain = contain

        self.__data = None
        
        # 物品用户字典,基于物品的协同过滤
        self.__itemUsers = None
        
        self.popularItemSet = None
        self.similarityMatrix = None
    

    def __getIntersection(self,a,b):
        '''得到a,b的交集
        '''
        intersection = list(a.keys() & b.keys())
        return intersection

    def consine(self,a,b):
        '''计算余弦相似度
        参数:
            a,b都是字典类型的单元素,根据其value计算余弦相似度
        return cos(a,b)
        '''
        sum_com = sum_a = sum_b = 0.0
        common = self.__getIntersection(a,b)
        for item in common:
            sum_com += a[item]
            sum_com += b[item]
        
        for item in a.keys():
            sum_a += a[item]
        for item in b.keys():
            sum_b += b[item]

        return sum_com / np.sqrt(sum_a * sum_b)
    
    def jaccard(self, a, b):
        '''计算jaccard相似度

        return 相似度
        '''
        sum_com = sum_a = sum_b = 0.0
        common = self.__getIntersection(a,b)
        for item in common:
            sum_com += a[item]
            sum_com += b[item]

        for item in a.keys():
            sum_a += a[item]
        for item in b.keys():
            sum_b += b[item]

        return sum_com / (sum_a + sum_b)

    def pearson(self, a, b):
        '''计算pearson相似度
        return 相似度
        '''
        common = self.__getIntersection(a,b)
        
        n = len(common)
        # 没有共同之处
        if len(common) == 0: 
            return 1
        # 对所有偏好求和
        sum_a = np.sum([a[item] for item in common])
        sum_b = np.sum([b[item] for item in common])

        # 求平方和
        sumSq_a = np.sum([a[item]*a[item] for item in common])
        sumSq_b = np.sum([b[item]*b[item] for item in common])

        # 求乘积之和
        pSum = np.sum([a[item]*b[item] for item in common])

        # 计算pearson值
        num = pSum - (sum_a*sum_b/n)
        den = np.sqrt((sumSq_a-sum_a*sum_a/n) * (sumSq_b-sum_b*sum_b/n))
        if den == 0: return 0
        return num / den


    def __recall(self,recomAns, items):
        '''
        get Recall
        真正购买的物品在推荐的物品中命中个数 / 真正购买的个数
        '''
        hit = 0
        all = 0

        for i in range(len(items)):
            all += len(items[i])
            for recomit in recomAns[i]:
                if recomit in items[i]:
                    hit +=1
                        
        return 1.0 * hit / all


    def __precision(self,recomAns, items):
        '''
        get Precision
        推荐的物品在真正购买的物品中的命中的个数 / 推荐的个数
        参数:
            都是列表
        '''
        hit = 0
        all = 0

        for i in range(len(items)):
            all += len(recomAns[i])
            for recomit in recomAns[i]:
                if recomit in items[i]:
                    hit +=1
            
        return 1.0 * hit / all

    def __F1Scoure(self,recomAns, items):
        """
        F1分数=2*recall*precision / (recall+precision)
        """
        recall = self.__recall(recomAns,items)
        precision = self.__precision(recomAns,items)

        return 2*recall*precision / (recall+precision);
        

    def __check_data(self,data):
        '''检查数据格式,是否满足条件

        '''
        if len(data) < 1:
            raise ValueError("数据长度为0")

        if not isinstance(data,dict):
            raise ValueError("数据类型不满足条件,请确定数据格式是参数要求的字典格式.")
        
        # 如果没有给出评分,进进行评分,默认评分是1
        firstData = None
        for key in data.keys():
            firstData = data[key]
            break

        if not isinstance(firstData,dict):
            newdata = {}
            for key in data.keys():
                itemScore = {}
                for item in data[key]:
                    itemScore[item] = 1
                newdata[key] = itemScore

            data = newdata
        return data

    
    def __item2user(self,data):
        '''基用户物品的字典转化为基于物品用户的字典

        '''
        itemUser = {}
        for user in data.keys():
            for item in data[user].keys():
                if item not in itemUser.keys():
                    itemUser[item] = {}
                itemUser[item][user] = data[user][item]

        return itemUser


    def fit(self,data,check_input=True):
        '''建立推荐系统
        参数:
            data: 字典类型,用户->物品 字典,
                [key, values] --> [usersID, {物品1:评分1,物品2:评分2,...,} ]
                [key, values] --> [usersID, [物品1,,物品2,...,] ] 如果没有分数数据,默认是1,
                建议保证物品和用户的唯一性
        return 
            self
        '''
        if check_input == True:
            data = self.__check_data(data)
        
        self.__data = data

        similarityM = {}

        if self.__similarity == 'pearson':
            simFunc = self.pearson
        elif self.__similarity == 'jaccard':
            simFunc = self.jaccard
        else: # 'cosine'
            simFunc = self.consine

        # 得到热门物品
        itemsdict = {}    
        for userA in data.keys():
            # 得到前20个最热门的物品,这里根据物品的评分总和尽心排序
            for item in data[userA].keys():
                if item not in itemsdict.keys():
                    itemsdict[item] = data[userA][item]
                else:
                    itemsdict[item] += data[userA][item]


        if self.__criterion == 'user':
            # 计算所有用户之间相似度,(可以使用倒排索引表加快计算速度,亲自测试可以提高无数倍)
            for userA in data.keys():
                tmp_sim = {}
                for userB in data.keys():
                    if userA != userB:
                        tmp_sim[userB] = simFunc(data[userA],data[userB])

                
                similarityM[userA] = tmp_sim
        
        else : # 'item',基于物品的协同过滤算法,转化数据结构
            data = self.__item2user(data)

            for itemA in data.keys():
                tmp_sim = {}
                for itemB in data.keys():
                    if itemA != itemB:
                        tmp_sim[itemB] = simFunc(data[itemA],data[itemB])

                
                similarityM[itemA] = tmp_sim


        self.similarityMatrix = similarityM

        # 得到最热门的100物品
        sortItem = sorted(itemsdict.items(),key=lambda x:x[1],reverse=True)
        popularItem = [sortItem[i][0] for i in range(min(100,len(sortItem)))]
        self.popularItemSet = popularItem

        return self

    
    def __predictOne(self,user):
        """给用户推荐与它相关的物品,根据给定的参数.
            参数是一个用户,
        """
        # 如果当前用户没有相似的用户,直接返回最热门的物品
        if user not in self.__data.keys():
            recomAns = [(d,-1) for d in self.popularItemSet[:min(self.__K,len(self.popularItemSet))]]

            return recomAns
    
        recommendItem = {}        

        if self.__criterion == 'user':
            #根据给定的参数得到这个用户的相似度列表
            similist = self.similarityMatrix[user]

            similist_sort = sorted(similist.items(),key=lambda x:x[1],reverse=True)
            n = min(self.__N, len(similist_sort))
            # 根据相似度最高的N的人,计算这N的人的对物品的评分总和,这里不推荐自己买过的物品
            
            for d in similist_sort[:n]:
                if d[0] in self.__data.keys():
                    for item in self.__data[d[0]].keys():
                        if item not in recommendItem.keys():
                            recommendItem[item] = self.similarityMatrix[user][d[0]] * self.__data[d[0]][item]
                        else:
                            recommendItem[item] += self.similarityMatrix[user][d[0]] * self.__data[d[0]][item]
    
        else: # 基于物品的协同过滤
            # 根据推荐用户的过去买过物品进行推荐,推荐与之前相似度最高的物品
            itemset = self.__data[user]
            for itHost in itemset.keys():
                for it in self.similarityMatrix[itHost].keys():
                    if it not in recommendItem.keys():
                        recommendItem[it] = 0
                    recommendItem[it] += self.similarityMatrix[itHost][it]
        

        sortRecom = sorted(recommendItem.items(),key=lambda x:x[1],reverse=True)
            
        # 不推荐用户自己曾经买过的物品,默认是不推荐的过去买过的物品
        if self.__contain == False:
            for it in sortRecom:
                if it[0] in self.__data[user].keys():
                    sortRecom.remove(it)


        # 推荐的物品个数不组K个,推荐热门物品,得分用-1表示,
        if len(sortRecom) < self.__K:
            tmp = [d[0] for d in sortRecom]
            for item in self.popularItemSet:
                if item not in tmp:
                    sortRecom.append((item,-1))
                if len(sortRecom) >= self.__K:
                    break
        k = min(len(sortRecom),self.__K)
        return sortRecom[:k]

    def predict(self,userlist):
        '''推荐物品,
        参数:
            userlist: 用户的列表数组

        '''
        recommendResult = []
        for user in userlist:
            recommendResult.append(self.__predictOne(user))
        return recommendResult

    def source(self,userSet,itemSet):
        '''根据召回率和准确率和F1的值评价模型的性能

        return 召回率,准确率,F1的值
        '''
        recommendResult = self.__predict(userSet)

        recomAns = []
        for its in recommendResult:
            tmp = [d[0] for d in its]
            recomAns.append(tmp)
        recall = self.__recall(recomAns,itemSet)
        precision = self.__precision(recomAns,itemSet)
        f1scoure = self.__F1Scoure(recomAns,itemSet)
        print("recall = ",recall)
        print("precision = ",precision)
        print("f1scoure = ",f1scoure)

        return recall, precision, f1scoure
