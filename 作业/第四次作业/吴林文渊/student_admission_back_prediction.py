import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#csv非常好用！！！！
data=pd.read_csv(r'F:\项目架构\2 神经网络\2自己动手实现第一个神经网络\student_data.csv')

#one-hot编码
def one_hot():
    rank_hot=pd.get_dummies(data['rank'],prefix='rank')
    one_hot_data=pd.concat([data,rank_hot],axis=1)
    one_hot_data=one_hot_data.drop('rank',axis=1)
    return one_hot_data

#缩放数据
def scaling_data(one_hot_data):
    processed_data=one_hot_data[:]
    processed_data['gre']=processed_data['gre']/800
    processed_data['gpa']=processed_data['gpa']/4
    return processed_data

#分离数据
def train_test_data(processed_data):
    sample=np.random.choice(processed_data.index,size=int(len(processed_data)*0.9))
    train_data,test_data=processed_data.iloc[sample],processed_data.drop(sample)

    train_feature=train_data.drop('admit',axis=1)
    train_target=train_data['admit']

    test_feature=test_data.drop('admit',axis=1)
    test_target=test_data['admit']

    return train_feature,train_target,test_feature,test_target

#激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#激活函数的导数
def sigmoid_prime(y):
    return y*(1-y)

#误差公式E
def errorE(y, output):
    return (output - y) ** 2

def main():
    #数据准备阶段
    one_hot_data=one_hot()
    processed_data=scaling_data(one_hot_data)
    train_feature, train_target, test_feature, test_target=train_test_data(processed_data)

    #神经网络层数 与后面权重的数量  以及反向传播公式的迭代次数有关
    n_hidden=2

    #权重 以及迭代的次数
    epochs = 1000
    learn_rate = 0.2
    n1, m1 = train_feature.shape
    weights_input_hidden = np.random.normal(scale=1 / n1 ** .5,size=(m1,6))
    weights_output_hidden=np.random.normal(scale=1 / n1 ** .5,size=(6))
    #两个网络层的权重行列数要匹配

    #初始化权重改变量
    w_input_hidden = np.zeros(weights_input_hidden.shape)
    w_output_hidden = np.zeros(weights_output_hidden.shape)
    #训练
    for e in range(epochs):
        #每轮累计下来对权重的改变量和误差
        loss=0
        for x,y in zip(train_feature.values,train_target.values):
            #隐藏层的输入与输出
            hidden_input=np.dot(x,weights_input_hidden)
            hidden_output=sigmoid(hidden_input)

            #输出层的输入与输出
            output_input=np.dot(hidden_output,weights_output_hidden)
            output=sigmoid(output_input)

            #开始把数学公式变为代码
            error=y-output

            #输出层
            output_error=error*sigmoid_prime(output)

            #隐藏层
            hidden_error=output_error*weights_output_hidden*sigmoid_prime(hidden_output)

            #权重改变量
            w_input_hidden=w_input_hidden+learn_rate*hidden_error*x[:,None]
            w_output_hidden=w_output_hidden+learn_rate*output_error*hidden_output
            loss=loss+errorE(y,output)

        weights_input_hidden=w_input_hidden+w_input_hidden/n1
        weights_output_hidden=w_output_hidden+weights_output_hidden/n1

        if e%100==0:
            loss=loss/n1
            print(e,'==========')
            print('loss=',loss)

    hidden=sigmoid(np.dot(test_feature.values,weights_input_hidden))
    out=sigmoid(np.dot(hidden,weights_output_hidden))
    prediction=out>0.5
    accuracy=np.mean(prediction==test_target.values)
    print(accuracy)














main()