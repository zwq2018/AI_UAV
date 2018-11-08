import numpy as np
"""
@description: 使用numpy多维数组对于矩阵运算的支持，简单的实现3层前向神经网络；
    所谓前向(forward)，是指处理过程是由输入层到输出层；
    这里定义的3层，是指有实质权重和偏置值的层数(可以理解为神经网络节点图中的线的层数)，
                  即这里包含两个隐藏层。
    从神经元个数上来看，定义输入层(第0层)为一维数组，包含两个元素(x1,x2)；第1层(隐藏层)
    包含3个神经元a1(1),a2(1),a3(1)；第2层(隐藏层)包含2个神经元，a1(2),a2(2)；输出层(第3层)
    包含两个输出值y1,y2。
    本程序中，隐藏层的激活函数使用sigmoid函数，而输出层激活函数的选择有两种：
              1. 使用恒等函数(即输出值等于输入值)[一般用于回归问题];
              2. 使用softmax函数(含指数函数因子)[一般用于分类问题]；
                 softmax函数之所以能够处理分类问题，是因为它具有所有输出值的总和为1的特性，这样可以将它的输出与概率(统计)结合起来，
                 用概率的思想去解释得到的结果；一般输出值最大的拥有最大的概率，即这个神经元所对应的类别有很大概率是给定输入值所在的类别。
    本程序是人工设定每一层权重和偏置值，没有涉及到神经网络的自动学习。
    该3层前向神经网络的定义和信号传递图在一并提交的文档中。
"""


"""定义激活函数sigmoid()，对输入信号的加权和(包括权重和偏置)进行转换，从而得到输出值；用于隐藏层"""
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


"""定义激活函数identity_function(),其转换是使输出值等于输入值；用于输出层"""
def identity_function(x):
    return x


"""定义激活函数softmax(),为防止溢出问题，这里采用softmax的改进型[各个输入元素减去输入元素的最大者]"""
def softmax(a):
    c = np.max(a)    # c为输入信号a中的最大者
    exp_a = np.exp(a - c)   # 得到输入元素减去最大值的指数值
    sum_exp_a = np.sum(exp_a)   #   将所有输入元素的指数值加和
    y = exp_a / sum_exp_a     #  得到输出值数组

    return y



"""神经网络的初始化函数，用于人工设置每一层权重W和偏置b的值，并将它们保存在字典中"""
def init_neural_network():

    neural_network = {}   # 字典对象，保存整个神经网络的权重和偏置

    # 设置第一层的权重矩阵W1和偏置数组b1
    neural_network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    neural_network['b1'] = np.array([0.1, 0.2, 0.3])

    # 设置第二层的权重矩阵W2和偏置数组b2
    neural_network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    neural_network['b2'] = np.array([0.1, 0.2])

    # 设置第三层的权重矩阵W3和偏置数组b3
    neural_network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    neural_network['b3'] = np.array([0.1, 0.2])

    return neural_network


"""3层前向神经网络的简单实现，接收输入信号x的numpy数组，使用softmax函数作为输出层的激活函数"""
def three_forward_network_classify(x):
    # 取得每一层的权重矩阵和偏置数组
    neural_network = init_neural_network()
    W1, W2, W3 = neural_network['W1'], neural_network['W2'], neural_network['W3']
    b1, b2, b3 = neural_network['b1'], neural_network['b2'], neural_network['b3']

    # 计算第一层的加权和A1和输出值Z1
    A1 = np.dot(x, W1) + b1
    Z1 = sigmoid(A1)

    # 计算第二层的加权和A2和输出值Z2
    A2 = np.dot(Z1, W2) + b2   # 这里的输入一定是上一层的输出Z1
    Z2 = sigmoid(A2)

    # 计算输出层的加权和A1和输出值Y
    A3 = np.dot(Z2, W3) + b3   # 这里的输入一定是上一层的输出Z2
    Y = softmax(A3)     # 使用softmax作为激活函数

    return Y


"""3层前向神经网络的简单实现，接收输入信号x的numpy数组，使用恒等函数作为输出层的激活函数"""
def three_forward_network_regression(x):
    # 取得每一层的权重矩阵和偏置数组
    neural_network = init_neural_network()
    W1, W2, W3 = neural_network['W1'], neural_network['W2'], neural_network['W3']
    b1, b2, b3 = neural_network['b1'], neural_network['b2'], neural_network['b3']

    # 计算第一层的加权和A1和输出值Z1
    A1 = np.dot(x, W1) + b1
    Z1 = sigmoid(A1)

    # 计算第二层的加权和A2和输出值Z2
    A2 = np.dot(Z1, W2) + b2   # 这里的输入一定是上一层的输出Z1
    Z2 = sigmoid(A2)

    # 计算输出层的加权和A1和输出值Y
    A3 = np.dot(Z2, W3) + b3   # 这里的输入一定是上一层的输出Z2
    Y = identity_function(A3)

    return Y


"""测试函数，给定输入信号X的值"""
def _main():
    X = np.array([1.0, 0.5])

    Y_classify = three_forward_network_classify(X)   # 得到输出值数组
    print("给定输入信号x1=1.0,x2=0.5,输出层激活函数为softmax，该3层前向神经网络的输出值为：")
    print(Y_classify)
    print()

    Y_regression = three_forward_network_regression(X)  # 得到输出值数组
    print("给定输入信号x1=1.0,x2=0.5,输出层激活函数为恒等函数，该3层前向神经网络的输出值为：")
    print(Y_regression)


_main()
