import numpy as np
import matplotlib.pyplot as plt
import xlrd
#最简单的感知器模型,二维特征值，线性模型，表示训练的过程
# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(2)


def stepFunction(t):  #0-1阶跃函数
    if t >= 0:
        return 1
    return 0


def prediction(X1,X2, W, b):
    return stepFunction((np.matmul([X1,X2], W) + b)[0])#matmul是矩阵乘法，X*W


# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X1,X2, y, W, b, learn_rate = 0.01):
    for i in range(len(X1)):
        y_hat = prediction(X1[i],X2[i],W,b) #根据当前的w计算预测值y
        if y[i]-y_hat == 1: #预测值偏小则调整w值 增加
            W[0] += X1[i]*learn_rate
            W[1] += X2[i]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1:   #预测值偏大则减小w
            W[0] -= X1[i]*learn_rate
            W[1] -= X2[i]*learn_rate
            b -= learn_rate
    return W, b


# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X1,X2, y,learn_rate=0.01, num_epochs=100):
    x_min, x_max = min(X1), max(X1)
    y_min, y_max = min(X2), max(X2)
    W = np.array(np.random.rand(2, 1)) #w初始值用随机数 是2*1的矩阵
    b = np.random.rand(1)[0] + x_max #b初值也是随机数
    # These are the solution lines that get plotted below.

    boundary_lines = []  #分界线   Y=W1X1+W2*X2+b,分界线是Y=0；则W1X1+W2*X2+b=0；X2=-W1/W2 *X1-B/W2
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X1,X2, y, W, b, learn_rate)
        plt.scatter(X1, X2, c=y)  # 颜色由标签定
        boundary_lines.append((-W[0] / W[1], -b / W[1]))
        xx=np.arange(0,2.0,0.01)
        yy=-W[0] / W[1] *xx-(b / W[1])
        plt.plot(xx,yy)
        plt.show()
    return boundary_lines


if __name__ == '__main__':
    data = xlrd.open_workbook(r'C:\Users\zhangwenqi\Desktop\神经网络test\第一课神经网络\第一章 神经网络介绍\code\data.xls')  # 打开xls文件
    table = data.sheets()[0]  # 打开第一张表
    X1=table.col_values(0)
    X2 =table.col_values(1)
    y=table.col_values(2)#输出只含0,1元素的一维数组,长度为100,为标签
    plt.figure(10*10)
    plt.scatter(X1, X2,c=y)  #颜色由标签定
    plt.show()



    boundary_lines=trainPerceptronAlgorithm(X1,X2,y,learn_rate=0.01, num_epochs=25)


