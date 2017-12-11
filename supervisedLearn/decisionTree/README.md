# decisionTree(决策树)
    最经典的三种方法,分别是ID3,C4.5和CART.下面介绍三种算法.
## ID3算法
    ID3算法最早由Ross Qulinlan发明,用来基于数据产生一个决策树.并且是C4.5的前身,广泛用于机器学习和自然语言处理方向. 
    ID3算法,其实道理很简单,对于给定的数据集合S和其所有的特征features,每次通过确定的方法(信息增益或者信息熵)选择一个最好的特征,将数据分成多个子集,在其子集上递归操作,知道不能选择特征,即不同分割.
    递归停止的条件如下几种:
        1. 子集中的每一个元素都属于同一个类(+ or -),因此这个节点被标记为叶子节点,这个类作为这个叶子节点的值.
        2. 没有特征被选择,选择子集中最多的类别作为这个子节点的标记值.
        3. 这个子集中没有元素,用其父节点最多的类标记这个节点,并作为叶子节点.
    总结:
        1. 计算每一个特征的信息熵,
        2. 使用最小的信息熵或者最大的信息增益对应的特征,将数据集S分成的K个子集,K是特征的类型个数.
        3. 根据这个特征做一个决策树.
        4. 在子集中低估以上步骤.
    伪代码:
        ID3(example, Target_Attribute, Attributes)
            create a root node for tree
            if all example are positive, Return sigle-node tree Root, with label = '+',(全是正样例)
            if all example are negetive, Return sigle-node tree Root, with label = '-',(全是负样例)
            if number of predicting attributes is empty,then return single node tree Root, with label = most common value of the target attribute in the example.(没有特征选择,返回最多的label)
            Other Begin:
                A <- 分类这些样例的最好特征.
                决策树 Root = A.
                FOR each possible value vi of A,
                    添加树root的子节点,使得所有的A = vi
                    let Example(vi) 作为一个子集继续运行
                    IF Example(vi) is Empty:
                        这个分支作为一个叶子节点,标记label = 自子集中出现最多的label.
                    else 递归调用, ID3(Example(vi), Target_Attribute, Attributes - {A})
            End
            return Root
    ID3算法不是最优的算法,它使用贪心的方法来决策,其实可以通过回溯的方法得到最优的树,但是这样的代价也是很大的.ID3算法能都很好的拟合训练数据,就可能会过拟合,为了防止过拟合,小的决策树一般会优于大的决策树.ID3算法很难使用在连续的数据上,可以先将连续数据等分成离散的数据在使用(但是不推荐).
    ID3的评价标准之信息熵:
        数据集S的熵(Entropy) 
$$ H(s) =  -\sum_i^n-p(x)log_2p(x) $$

        where S是当前的数据集,X是所有的类的集合,x是其中一个类,p  (x)是类x的个数所占的比例.H(S)越小越好,在ID3中计算熵,需要计算剩下的所有的特征的熵,选择最小的熵对应的特征.

    ID3的评价标准之信息增益:
        信息增益是(熵之间的差值,IG(A) = 数据集S的熵H(S) - 根据特征A分割数据后的所有子集熵的和.换句话说就是根据A分割数据后,S的不确定性被降低多少.
$$ IG(A) = H(S) - \sum_{t \in T}p(t)H(t) $$

    H(S): S的信息熵.
    T: 根据A分割的所有的子集.
    p(t): 子集t的个数占所有集合S的比例.
    H(t): 子集t的信息熵.

    在ID3中,信息增益最大的对应的特征,作为这个迭代划分的特征,作为一个节点.

## C4.5算法
    C4.5是ID3的改进,使用信息增益比来进行特征的选择,信息增益比 = 信息增益 /( 信息熵.

$$ IG_RATE(A) = \frac {IG(A)} {H(S)} $$

## CART算法
