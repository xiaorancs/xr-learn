# LogisticRegression
一个简单但是高效的分类器,这里给出二分类问题的代码,同样支撑sklearn中常用的接口.
## 参数
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

使用平均梯度下降进行计算.

## example:
```
import LogisticRegression
lr = LogisticRegression.LogisticRegerssion()
x = [[1,2],[1,3],[1,4],[2,1],[3,1]]
y = [1,1,1,0,0]

lr.fit(x,y)
lr.predict([[1,5],[1,6]])
lr.predict_proba([7,1])

lr.source([[1,5],[1,6],[6,1]],[1,1,0])


```