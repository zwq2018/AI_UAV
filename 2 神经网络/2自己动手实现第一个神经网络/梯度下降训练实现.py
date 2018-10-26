#实现梯度下降算法
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Some helper functions for plotting and drawing lines

def plot_points(X, y):#将两个类别的点按照标签绘图
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'blue', edgecolor = 'k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'red', edgecolor = 'k')

def display(m, b, color='g--'):#绘制当前的分割直线
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    x = np.arange(-10, 10, 0.1)
    plt.plot(x, m*x+b, color)



# Activation (sigmoid) function
def sigmoid(x):#激活函数  采用sigmod函数
    return 1 / (1 + np.exp(-x))
#实现计算sigmoid(w1*x2+w2*x2+b)，即计算输出预测值
def output_formula(features, weights, bias):
    return sigmoid(np.dot(features, weights) + bias)  #dot是矩阵乘法，feature是2*n矩阵，weight是n*1

def error_formula(y, output):#计算误差函数 针对每一个yi计算
    return - y*np.log(output) - (1 - y) * np.log(1-output)

def update_weights(x, y, weights, bias, learnrate):#权重更新方法，根据梯度下降法来更新
    output = output_formula(x, weights, bias)
    d_error = -(y - output)
    weights -= learnrate * d_error * x
    bias -= learnrate * d_error
    return weights, bias




#训练函数，用于训练分界线
def train(features, targets, epochs, learnrate, graph_lines=False):
    errors = []
    n_records, n_features = features.shape#n_records=100,n_features=2
    last_loss = None
    weights = np.random.normal(scale=1 / n_features ** .5, size=n_features) #初始值用随机数生成权重 2*1
    bias = 0
    display(-weights[0] / weights[1], -bias / weights[1])  # 画当前求解出来的分界线

    for e in range(epochs): #迭代1000次
        del_w = np.zeros(weights.shape)
        for x, y in zip(features, targets):#通过zip拉锁函数将X与y的每个点结合起来
            output = output_formula(x, weights, bias) #计算输出预测值yi 其中x是1*2，weight是2*1
            error = error_formula(y, output)#计算每一个yi的误差
            weights, bias = update_weights(x, y, weights, bias, learnrate)
        print(weights,bias)
        print(e)#注意 每次迭代里都对xi即100组数进行计算都更新了权重，即更新了100*迭代次数次，每次迭代都是以上次的结果重新计算100组数
        # Printing out the log-loss error on the training set
        out = output_formula(features, weights, bias)#计算迭代后的预测值，这里feature是n*2,weight是2*1，out是n*1的一列预测值
        loss = np.mean(error_formula(targets, out))#对每个预测值的误差做算术平均
        errors.append(loss)
        if e % (epochs / 10) == 0:
            print("\n========== Epoch", e, "==========")
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            predictions = out > 0.5
            accuracy = np.mean(predictions == targets)
            print("Accuracy: ", accuracy)
        if graph_lines :#and e % (epochs / 100) == 0
            display(-weights[0] / weights[1], -bias / weights[1])#画当前求解出来的分界线

    # Plotting the solution boundary
    plt.title("Solution boundary")
    display(-weights[0] / weights[1], -bias / weights[1], 'black')#画最后一根求解出来的分界线

    # Plotting the data
    plot_points(features, targets)
    plt.show()

    # Plotting the error
    plt.title("Error Plot")
    plt.xlabel('Number of epochs')
    plt.ylabel('Error')
    plt.plot(errors)
    plt.show()



if __name__ == '__main__':

    np.random.seed(44)

    epochs = 100
    learnrate = 0.01

    data = pd.read_csv('C:/Users/zhangwenqi/Desktop/神经网络test/deep-learning-master/deep-learning-master/gradient-descent/data.csv', header=None)
    X = np.array(data[[0,1]])  #data是一个dataframe,X是一个100*2的一个矩阵数组
    y = np.array(data[2])#y是100*1的一个数组
    plot_points(X,y)
    plt.show()
    train(X, y, epochs, learnrate, True)


