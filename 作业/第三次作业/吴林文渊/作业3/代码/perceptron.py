#二分类感知器预测
import numpy as np
import xlrd
np.random.seed(44)#这里的可预测性是指相同的种子（seed值）所产生的随机数是相同的
import matplotlib.pyplot as plt
def stepFunction(t):#二分类使用的阶跃函数
    if t>=0:
        return 1
    else:
        return 0

def prediction(X,W,b):
    return stepFunction((np.matmul(X,W)+b))#矩阵乘法

def perceptronStep(X,y,W,b,learn_rate):#X 输入 y 标签  W 权重  b 偏差
    for i in range(len(X)):
        y_hat=prediction(X[i],W,b)#m每一行两个数字组成的数组进行矩阵乘法
        if y[i]-y_hat==1:#应该是1但是分成了0  权重偏小
            W[0]=W[0]+X[i][0]*learn_rate
            W[1]=W[1]+X[i][1]*learn_rate
            #W=W+X[i]*learn_rate W是列二维数组  X是行二维数组  不能直接乘
            b=b+learn_rate
        elif y[i]-y_hat==-1:#应该是0但是分成了1  权重偏大
            W[0]=W[0]-X[i][0]*learn_rate
            W[1]=W[1]-X[i][1]*learn_rate
            b=b-learn_rate
    #如果分正确就什么也不做
    return W,b#返回更新的权值和偏差

def data():#准备输入数据
    X=[]
    file=xlrd.open_workbook(r'F:\项目架构\2 神经网络\2自己动手实现第一个神经网络\data.xls')
    table=file.sheets()[0]#表格里有三张表 有数据的只有第一张
    x1=table.col_values(0)#第一列
    x2=table.col_values(1)#第二列

    X.append(x1)
    X.append(x2)
    X = np.array(X).T#输入的二维数据每行两个数据
    y = table.col_values(2)#标签

    W=np.array(np.random.rand(2,1))#随机列二维数组
    print('W=',W)
    b = np.random.rand(1)[0] + max(x1)#随便分配一个

    return X,W,y,b,x1,x2

def main():
    X,W,y,b,x1,x2=data()
    learn_rate=0.05
    num_epochs=1000#训练1000次
    for i in range(0,num_epochs):
        W,b=perceptronStep(X,y,W,b,learn_rate)
    plt.scatter(x1, x2, c=y)#颜色由标签定
    xx = np.arange(0, 1.0, 0.01)
    yy = -W[0] / W[1] * xx - (b / W[1])#画直线
    plt.plot(xx,yy)
    plt.show()
    print(W[0],W[1])
    print(b)
main()









