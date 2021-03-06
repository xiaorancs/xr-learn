# 特征选择(feature select)
    特征选择是一种有效的预处理数据的方式，是解决维数灾难的一种方式，另一种方式是降维(PCA)。 
    去除不相关的特征会降低学习任务的难度。
在特征选择中有两种类型，无关特征和冗余特征
1. 无关特征
    与当前学习任务无关的额特征。

2. 冗余特征
    它们所包含的信息能够从其他特征推演出来，冗余特征会降低学习的难度，某些冗余特征恰好对应了完成学习任务所需的“中间概念”，这样的特征是有用的。

特征选择，我们可以进行所有特征的暴力组合，但是这样是不可以的，于是是基于贪心的特征选择，可以进行添加或者删除进行贪心形式的判断。其实决策树模型的生成过程就是一种特征选择的方式。

## 特征选择的方式
1. 过滤式（filter）
2. 包裹式（wrapper）
3. 嵌入式（embedding）

### 过滤式特征选择
    先对数据进行特征选择，然后在训练学习器，特征的选择过程和学习器没有关系。Relief（Relevant Feature）是一种著名的过滤式特征选择方法，该方法设计出一种“相关统计量”来度量特征的重要性。
    例如：给定数据集{ (x_1,y_1),(x_2,y_2),...,(x_m,y_m)}, 对每一个示例x_i, Relief先在x_i的同类样本中寻找其最近邻x_i,nh, 称为“猜中近邻”，在从x_i的异类样本中寻找其最近邻x_i,nm, 称为“猜错近邻”，然后，相关统计量对应属性j的分量为：
$$ l ^ {j} $$