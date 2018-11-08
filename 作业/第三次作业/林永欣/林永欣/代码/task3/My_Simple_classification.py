import numpy as np
import matplotlib.pyplot as plt
import xlrd


def step_function(t):  # 0-1阶跃函数
    if t >= 0:
        return 1
    return 0


def prediction(x1, x2, w, b):
    return step_function((np.matmul([x1, x2], w) + b)[0])  # matmul是矩阵乘法，X*w


def perceptron_step(x1, x2, y, w, b, learn_rate):
    for i in range(len(x1)):
        y_hat = prediction(x1[i], x2[i], w, b)  # 根据当前的w计算预测值y
        if y[i] - y_hat == 1:  # 预测值偏小则调整w值 增加
            w[0] += x1[i] * learn_rate
            w[1] += x2[i] * learn_rate
            b += learn_rate
        elif y[i] - y_hat == -1:  # 预测值偏大则减小w
            w[0] -= x1[i] * learn_rate
            w[1] -= x2[i] * learn_rate
            b -= learn_rate
    return w, b


'''感知器算法训练'''


def train_algorithm(x1, x2, y, learn_rate, num_epochs):
    np.random.seed(2)  # 设置生成的随即数都相同
    w = np.array(np.random.rand(2, 1))  # w初始值用随机数 是2*1的矩阵
    b = np.random.rand(1)[0]  # b初值也是随机数

    boundary_lines = []  # 分界线   Y=w1x1+w2*x2+b,分界线是Y=0；则w1x1+w2*x2+b=0；x2=-w1/w2 *x1-B/w2
    for i in range(num_epochs):
        w, b = perceptron_step(x1, x2, y, w, b, learn_rate)  # 感知器
        plt.scatter(x1, x2, c=y)  # 颜色由标签定
        boundary_lines.append((-w[0] / w[1], -b / w[1]))  # y=wx+b，这里的y=x2，x=x1,b=-b/w[1],w=-w[0]/w[1]
        xx = np.arange(0, 2.0, 0.1)  # 0到2，间隔是0.1
        yy = -w[0] / w[1] * xx - (b / w[1])
        plt.plot(xx, yy)
        plt.show()


def main():
    # file_path = input('请输入数据文件所在路径及文件名: ')
    file_path = r'F:\项目架构\2 神经网络\2自己动手实现第一个神经网络\data.xls'
    datafile = xlrd.open_workbook(file_path)  # 打开xls文件
    # sheet_name = input('请输入要获取的Sheet: ')
    sheet_name = u'Sheet1'
    table = datafile.sheet_by_name(sheet_name)  # 通过名字获取的方式打开第一张表
    x1 = table.col_values(0)  # 获取第0列数据
    x2 = table.col_values(1)  # 获取第1列数据
    y = table.col_values(2)  # 获取第三列数据，第三列是输出的数据，是我们需要的标签
    plt.figure(20 * 20)  # 定义一个20*20的图y框
    plt.scatter(x1, x2, c=y)  # 创建散点图，颜色由标签定
    plt.show()
    learn_rate = 0.01  # 学习速率，可修改
    num_epochs = 100  # 训练迭代次数，可修改
    train_algorithm(x1, x2, y, learn_rate, num_epochs)


if __name__ == '__main__':
    main()

