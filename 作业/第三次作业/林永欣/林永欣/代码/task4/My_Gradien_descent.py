# 实现梯度下降算法
import matplotlib.pyplot as plt
import numpy as np
import xlrd


# 显示出所有的误差算术平均值
def errors_show(errors):
    plt.title("Error Plot")
    plt.xlabel('Number of num_epochs')
    plt.ylabel('Error')
    plt.plot(errors)
    plt.show()


# 绘图和绘制线条的辅助函数,将两个类别的点按照标签绘图
def plot_points(x, y):
    admitted = x[np.argwhere(y == 1)]
    rejected = x[np.argwhere(y == 0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s=25, color='blue', edgecolor='k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s=25, color='red', edgecolor='k')


# 绘制当前的分离平面
def draw_separation_plane(weights, bias, color='g--'):
    w = -weights[0] / weights[1]  # y=W1x1+W2*x2+b,分界线是y=0；则W1x1+W2*x2+b=0；x2=-W1/W2 *x1-B/W2
    b = -bias / weights[1]
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    x = np.arange(0, 10, 0.1)
    plt.plot(x, w*x+b, color)


# 激活函数,采用sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 实现计算sigmoid(w1*x2+w2*x2+b)，即计算输出预测值
def output_predictive_value(x, weights, bias):
    return sigmoid(np.dot(x, weights) + bias)  # dot是矩阵乘法，x是2*n矩阵，weight是n*1


# 计算误差函数 针对每一个yi计算
def error_formula(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)


# 根据梯度下降法来更新权重
def update_weights(x, y, weights, bias, learn_rate):
    result = output_predictive_value(x, weights, bias)
    d_error = -(y - result)
    weights -= learn_rate * d_error * x
    bias -= learn_rate * d_error
    return weights, bias


# 训练分界线的算法
def train_algorithm(x, y, num_epochs, learn_rate):
    errors = []
    n_records, n_features = x.shape  # n_records=100,n_features=2
    weights = np.random.normal(scale=1 / n_features ** .5, size=n_features)  # 初始值用随机数生成权重 2*1
    bias = 0
    draw_separation_plane(weights, bias)  # 画当前求解出来的分界线,y=w1x1+w2*x2+b，简写为y=wx+b

    for i in range(num_epochs):  # 迭代num_epochs次
        for x1, y1 in zip(x, y):  # 通过zip函数将x与y的每个点结合起来
            weights, bias = update_weights(x1, y1, weights, bias, learn_rate)  # 更新权重
        result = output_predictive_value(x, weights, bias)  # 计算迭代更新后的预测值，这里x是n*2,weight是2*1，out是n*1的一列预测值
        loss = np.mean(error_formula(y, result))  # 对每个预测值的误差做算术平均——熵
        errors.append(loss)
        draw_separation_plane(weights, bias)  # 画当前求解出来的分界线

    # 显示图像
    plt.title("Last result")
    draw_separation_plane(weights, bias, 'red')  # 画最后一根求解出来的分界线,显示为红色
    plot_points(x, y)  # 把所有点都显示出来
    plt.show()

    # 画出最后的误差图
    errors_show(errors)


# 读文件数据，并生成原始散点图
def input_data():
    x = []
    file_path = r'F:\项目架构\2 神经网络\2自己动手实现第一个神经网络\data.xls'
    datafile = xlrd.open_workbook(file_path)  # 打开xls文件
    sheet_name = u'Sheet1'
    table = datafile.sheet_by_name(sheet_name)  # 通过名字获取的方式打开第一张表
    x1 = table.col_values(0)  # 获取第0列数据
    x2 = table.col_values(1)  # 获取第1列数据
    y = table.col_values(2)  # 获取第2列数据，第2列是输出的数据，是我们需要的标签

    x.append(x1)
    x.append(x2)
    x = np.array(x)

    x = np.array(x).T  # x是一个100*2的一个矩阵数组
    y = np.array(y)  # y是100*1的一个数组
    plot_points(x, y)
    plt.show()
    return x, y


def main():
    num_epochs = 100
    learn_rate = 0.01
    x, y = input_data()  # 读取xls文件的数据并转为numpy格式
    train_algorithm(x, y, num_epochs, learn_rate)  # 训练数据


if __name__ == '__main__':
    main()
