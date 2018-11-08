"""
@description: 该程序保存神经网络实现和学习中需要用到的函数，以方便调用
"""
import numpy as np

"""激活函数sigmoid()"""
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


"""激活函数softmax()"""
def softmax(x):
    if x.ndim == 2:
        x = x.T      # 如果x是二维，则需要对其进行转置
        x = x - np.max(x, axis=0)  # 溢出对策：根据softmax函数的特性，将每个x输入减去其中的最大值
        y = np.exp(x) / np.sum(np.exp(x))   # 对数组中的每个元素进行运算，得到每个元素的对应值数组

        return y.T    #  将结果数组转置回原来的形状

    x = x - np.max(x)   # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


"""激活函数identity_function()"""
def identity_function(x):
    return x


"""均方误差形式的损失函数"""
"""
    参数y和t均为numpy数组
    :param y: 表示神经网络的输出
    :param t: t表示监督数据[分类问题中为解标签]；
            t有两种表示形式：1.one-hot表示——只有正确解标签为1，其他标签为0；2.标签表示——各个解标签有不同值
    :return: 神经网络的输出值和训练数据中的实际值之间的均方误差
"""
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


"""交叉熵误差形式的损失函数"""
"""
   说明——该部分是mini-batch版(小批量训练数据)交叉熵误差的实现[注意最终要得到误差是每个数据误差的加权平均]
   两种情况——1.监督数据是one-hot-vector的情况（单分类：如一个图像上只有一个类别[是..与不是..的问题]）；只有正确解标签的tk为1，其余为0，则交叉熵误差为0，
                所以实际上只计算对应正确解标签所在索引的神经网络的输出的自然对数即可[可省略一些计算]
              2. 监督数据是n--hot的情况（多分类：如一个图像上存在两个类别[既有..又有..的问题]）；每一个数据的每种类别都单独计算交叉熵误差，将这些类别的误差加和，
                再加和所有数据的误差，然后求平均。
   问题——为了防止出现np.log(0)为负无限大-inf的情况，需要给y加上一个微小值delta
"""
def cross_entropy_error(y, t):
    delta = 1e-7   # 设定微小值
    if y.ndim == 1:  # ndim得到输出y的维度，其为1时表示求单个数据的交叉熵误差，需要改变数据的形状
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]   # 输出y的行数为选取的mini-batch的数据个数
    return -np.sum(t * np.log(y + delta)) / batch_size


"""基于数值微分的方法计算梯度[相对BP算法的特点：实现简单，但较费时间]"""
"""
   为对应形状为多维数组的权重参数W，这里与gradient_descent_math中的方法稍有不同，但本质一样。
"""
def numerical_gradient(f, x):
    h = 1e-4  # 0.0001  设定微小值
    grad = np.zeros_like(x)  # 梯度的形状和自变量x的形状相同

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])  # 设置在权重参数多维数组中的迭代器
    while not it.finished:
        idx = it.multi_index   # idx依次遍历x(W)中的每个元素，以对每个元素求梯度
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值
        it.iternext()     # 迭代器后移

    return grad