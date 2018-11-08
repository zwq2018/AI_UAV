import matplotlib.pyplot as plt
import numpy as np
import xlrd

"""按标签绘制数据点"""
def plot_points(X, y):
    admitted = X[np.argwhere(y == 1)]
    rejected = X[np.argwhere(y == 0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'blue', edgecolor = 'k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'red', edgecolor = 'k')

"""获取输入数据，并生成散点图"""
def GetInputData(sourcefile):
    """获取输入数据"""
    X = []
    data = xlrd.open_workbook(sourcefile)  # 打开xls文件
    table = data.sheets()[0]
    X1 = table.col_values(0)    #第一列数据：特征1
    X2 = table.col_values(1)    #第二列数据：特征2
    X.append(X1)
    X.append(X2)
    X = np.array(X).T   # 将X转换为100*2的矩阵
    Y = np.array(table.col_values(2))  # 第三列数据：数据点的标签
    """生成原始数据散点图"""
    plot_points(X, Y)
    return X, Y

"""绘制分离超平面"""
def MakeHyperplane(W, b, color = 'g--'):
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    x = np.arange(-10, 10, 0.1)
    y = -W[0] / W[1] * x - (b / W[1])
    plt.plot(x, y, color)

"""激活函数"""
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

"""权值更新"""
def weight_update(x, y, weights, bias, learn_rate):
    model_output = sigmoid(np.dot(x, weights) + bias)    # 当前模型输出
    error = model_output - y    # 当前模型误差
    weights -= learn_rate * error * x    # 梯度下降
    bias -= learn_rate * error
    return model_output, weights, bias

"""损失函数"""
def loss_formula(y, output):
    return - y * np.log(output) - (1 - y) * np.log(1- output)

"""误差跟踪：用于观测模型的误差变化，判断参数更新是否正确"""
def loss_trace(features, targets, weights, bias):
    out = sigmoid(np.dot(features, weights) + bias)
    loss = np.mean(loss_formula(targets, out))
    return loss

"""输出当前的模型误差"""
def loss_output(loss, last_loss , i):
    print("Epoch:", i)
    if last_loss and last_loss < loss:
        print("Train loss: ", loss, "  WARNING - Loss Increasing")
    else:
        print("Train loss: ", loss)
    print("=========")

sourcefile = r'F:\项目架构\2 神经网络\2自己动手实现第一个神经网络\data.xls'
epochs = 400
learn_rate = 0.01
X, Y = GetInputData(sourcefile)     # n_records=100,n_features=2
n_records, n_features = X.shape
weights = np.random.normal(scale = 1 / n_features ** .5, size = n_features)  # 初始值用随机数生成权重 2*1
bias = 0
last_loss = None    # 参数更新前的模型误差，用于模型的误差跟踪
"""模型训练"""
for i in range(epochs):
    for x_i, y_i in zip(X, Y):
        model_output, weights, bias = weight_update(x_i, y_i, weights, bias, learn_rate)
    if i%20 == 0:
        MakeHyperplane(weights, bias)
        """模型误差跟踪"""
        loss = loss_trace(X, Y, weights, bias)
        loss_output(loss, last_loss, i)
        last_loss = loss
"""绘制最终的分离超平面"""
MakeHyperplane(weights, bias, 'black')
plt.show()
