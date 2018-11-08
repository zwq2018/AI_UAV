import numpy as np
import matplotlib.pyplot as plt
import xlrd

"""获取输入数据，并生成散点图"""
def GetInputData(sourcefile):
    """获取输入数据"""
    data = xlrd.open_workbook(sourcefile)
    table = data.sheets()[0]
    X1 = table.col_values(0)  # 第一列数据：特征1
    X2 = table.col_values(1)  # 第二列数据：特征2
    Y = table.col_values(2)   # 第三列数据：每个点的标签
    """生成数据散点图"""
    plt.figure(10*10)
    plt.scatter(X1, X2, c = Y)  # 颜色由标签定
    return X1, X2, Y

"""激活函数"""
def StepFunction(x):
    if x >= 0:
        return 1
    return 0

"""感知机模型"""
def Perceptron(X1, X2, Y, W, b, learn_rate):
    for i in range(len(X1)):
        model_output = StepFunction((np.matmul([X1[i], X2[i]], W) + b)[0])    # 当前模型输出
        """预测值偏小则梯度上升，偏大则梯度下降"""
        # 对于直线y = w*x + b, y对w的偏导数为x，对b的偏导数为1，分别乘以学习率，即为每次迭代的步长
        # W和b的更新必须保持同步
        if Y[i] - model_output == 1:
            W[0] += X1[i] * learn_rate
            W[1] += X2[i] * learn_rate
            b += learn_rate
        elif Y[i] - model_output == -1:
            W[0] -= X1[i] * learn_rate
            W[1] -= X2[i] * learn_rate
            b -= learn_rate
    return W, b

"""根据感知机模型生成分离超平面"""
def MakeHyperplane(W, b):
    hyperplane = []
    hyperplane.append((-W[0] / W[1], -b / W[1]))
    x = np.arange(0, 2.0, 0.01)
    y = -W[0] / W[1] * x - (b / W[1])
    plt.plot(x, y)

sourcefile = r'F:\项目架构\2 神经网络\2自己动手实现第一个神经网络\data.xls'
X1, X2, Y = GetInputData(sourcefile)
W = np.array(np.random.rand(2, 1))  # 随机生成w和b的初始值，w为2*1的矩阵
b = np.random.rand(1)[0]
learn_rate = 0.01   #学习率
epoch = 30          #迭代次数
for i in range(epoch):
    W, b = Perceptron(X1, X2, Y, W, b, learn_rate)
"""绘制分离超平面"""
MakeHyperplane(W, b)
plt.show()



