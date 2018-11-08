import matplotlib.pyplot as plt
import numpy as np
import xlrd
import pandas as pd
np.random.seed(44)

def display(W,b,color='g--'):
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    xx = np.arange(0, 1.0, 0.01)
    yy = -W[0] / W[1] * xx - (b / W[1])  # 画直线
    plt.plot(xx,yy,color)

def sigmoid(x):#sigmoid 函数
    return 1.0/(1+np.exp(-x))

def output(X,W,b):#输出   依旧是一个点一个点来更新的
    return sigmoid(np.dot(X,W)+b)#  np.dot(2,3) 两个行向量的内积  都是行向量

def error(y,output):#误差函数 y如果不是np类型会报错 因为list 参数类型未知
    return -y*np.log(output)-(1-y)*np.log(1-output)

def update_w_b(X,W,b,y,learn_rate):#还是一个点一个点进行梯度下降法进行更新
    y_output=output(X,W,b)
    d_error=-(y-y_output)#标量
    W -= learn_rate * d_error * X
    b -= learn_rate * d_error
    return W,b

def data():#csv文件打不开 换为xls文件
    X = []
    file = xlrd.open_workbook(r'F:\项目架构\2 神经网络\2自己动手实现第一个神经网络\data.xls')
    table = file.sheets()[0]  # 表格里有三张表 有数据的只有第一张
    x1 = table.col_values(0)  # 第一列
    x2 = table.col_values(1)  # 第二列

    X.append(x1)
    X.append(x2)
    X = np.array(X).T  # 输入的二维数据每行两个数据
    y = np.array(table.col_values(2))  # 标签  X 是array类型  list 和 array 不能相乘
    #y=table.col_values(2)
    W=np.random.normal(scale=1 / 100** .5, size=2)#高斯分布（Gaussian Distribution）的概率密度函数  scale越小越瘦高
    b=0#随便分配一个
    return X,W,y,b,x1,x2

def main():
    X, W, y, b, x1, x2 = data()
    display(W,b)
    learn_rate = 0.01
    num_epochs = 100  # 训练100次
    for i in range(0,num_epochs):
        for x_t,y_t in zip(X,y):#这个地方出错   z=zip(X,y)不对
            W,b=update_w_b(x_t,W,b,y_t,learn_rate)
        print(i)
        y_output=output(X,W,b)
        print('y_output:',y_output)
        print('y_output_type',type(y_output))
        print('y_type:',type(y))
        print('y=',y)
        plt.scatter(x1, x2, c=y)  # 颜色由标签定
        display(W, b, 'k')
        plt.show()
        #loss=np.mean(error(y,y_output))
        #print('loss=',loss)

    print('W=',W)
    print('b=',b)
    plt.scatter(x1, x2, c=y)  # 颜色由标签定
    display(W,b,'k')
    plt.show()

main()