import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
#神经网络class
import bike_class as Network
#数据准备
import bike_data as bd
rides = pd.read_csv(r'F:\项目架构\作业\第五次作业\作业-曹力月\Predict_ofo\bike-sharing-dataset\hour.csv')

#损失函数r
def errorE(y, output):
    return np.mean((output - y) ** 2)

def start_train():
    # 数据准备阶段
    data = bd.one_hot(rides)
    scaled_features,data = bd.standard_data(data)
    test_features, test_targets, train_features, train_targets, val_features, val_targets = bd.data_test_train_val(data)

    #设置初始参数
    epoch=1000
    learn_rate=0.5
    hidden_nodes=10
    output_nodes=1
    input_nodes=test_features.shape[1]
    network = Network.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learn_rate)

    #保存loss
    loss={'train':[],'validation':[]}

    #开始训练
    for e in range(epoch):
        batch=np.random.choice(train_features.index,size=int(train_features.size/1000))
        x,y=train_features.ix[batch].values,train_targets.ix[batch]['cnt'].values

        #以随机数据进行初始化权重
        network.train(x,y[:None])

        #使用测试集对模型的迭代次数进行估计和判断
        train_loss=errorE((network.run(train_features).T),(train_targets['cnt'].values))
        val_loss=errorE((network.run(val_features).T),(val_targets['cnt'].values))

        loss['train'].append(train_loss)
        loss['validation'].append(val_loss)

    #显示损失值在训练中的下降过程
    plt.plot(loss['train'],label='Training loss')
    plt.plot(loss['validation'],label='Validation')
    plt.show()

    #预测值与目标值之间的差距  注意把缩放的数据恢复到原来的大小
    mean, std = scaled_features['cnt']
    predictions = network.run(test_features).T * std + mean

    plt.plot(predictions[0],label='p')
    plt.plot(((test_targets['cnt']*std+mean).values).T,label='t')

    plt.show()

start_train()

