# 感知机(perceptron)
我们一直都不知道这个模型是好还是坏,但是如果把它当做一个神经网络的一层,那就厉害了.首先从个人角度感性理解这个模型是什么?   
感知机:有一组数据(X,Y),Y = {+1,-1},我们希望找到一个函数F(x),是得到所有的数据都能正确分类.   
我们希望找到这样一个函数:F(x) = sign(WX+b).  
使得所有的训练数据都能正确分类.找到一个loss(x)函数,最小化其损失,来更新W和b.
1. 默认(W,b) = ([0,0,...,0],0)
2. 对弈数据(X_i,Y_i),F(X_i) = W_t * X_i + b_t, 
如果这是一个错误分类,会有Y_i * F(X_i) < 0. 此时更新:
 + W_t+1 = W_t + a * (Y_i * X_i)
 + b_t+1 = b_t + a * Y_i
3. a是学习率.

## 从感性到理论的推理
1. 要求训练集(X,Y)严格可分. X_i 是一个向量,Y_i = {+1, -1}
2. 使得WX + b = 0.
3. loss函数的选择,但是第一步选择误分类的总数,但是这不是W和b的连续可导函数,于是我们选择最小化所有点到平面的距离之和sum最小.
4. f(x) = sign(Wx+b)
5. loss(x) = L(W,b) = - sum{y_i * (w * x_i + b)}
6. 最小化loss(x),L(W,b).min L(W,b),得到W和b的梯度:
    + grad(W) = - sum (y_i * X_i)
    + grad(b) = - sum (y_i * b)
7. 更新W和b:
    + W_t+1 = W_t + a * grad(w)
    + b_t+1 = b_t + a * grad(b)
8. a是学习率.


## perceptron Code:
1. fit(X_train,y_train) 训练数据得到模型.
2. source(X,y) 验证集的得分,这里默认是正确率.
3. get_params() 得到参数
4. predict(X) 预测X的分类

# test
```
import perceptron

per = perceptron.Perceptron(verbose = 5)
X = [[1,0],[2,0],[2,1],[0,1],[0,2],[1,2],[1,3]]
y = [1,1,1,0,0,0,0]
X_vali = [[4,1],[1,5],[2,6]]
y_vali = [1,0,0]

per.fit(X,y)
per.source(X_vali,y_vali)
per.predict([[2,3],[5,1],[6,3]])
per.get_params()
```

