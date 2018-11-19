import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data_test=pd.read_csv('C:/Users/zhangwenqi/Desktop/data.csv')
data_test = data_test.sample(frac=1)#打乱所有的数据

def deal_data(data):#数据和标签分隔开
    feature=data.iloc[:,0:4]
    label=data['class']
    return feature,label

def one_hot(label):#编码
    class_hot=pd.get_dummies(label,prefix='class')
    return class_hot

def standard_data(data):#缩放数据
    min1=np.min(data['feature1'])
    max1=np.max(data['feature1'])
    data['feature1']=(data.iloc[:,0]-min1)/(max1-min1)

    min2=np.min(data['feature2'])
    max2=np.max(data['feature2'])
    data['feature2']=(data.iloc[:,1]-min2)/(max2-min2)

    min3=np.min(data['feature3'])
    max3=np.max(data['feature3'])
    data['feature3']=(data.iloc[:,2]-min3)/(max3-min3)

    min4=np.min(data['feature4'])
    max4=np.max(data['feature4'])
    data['feature4'] = (data.iloc[:,3] - min4) / (max4 - min4)

    return data

def data_test_train(feature,label):#分离数据集
    train_feature=feature.iloc[0:int(len(feature)*0.9)]
    train_target=label.iloc[0:int(len(label)*0.9)]
    test_feature=feature.iloc[int(len(feature)*0.9)+1:]
    test_target=label.iloc[int(len(label)*0.9)+1:]

    return train_feature,train_target,test_feature,test_target

def softmax(x):#激活函数 没问题
    z=np.exp(x)
    s=z/sum(z)
    return s

def loss_derivative(y,y_hat):#多分类交叉熵
    loss=-np.dot(y,np.log(y_hat))#内积
    return loss

def error(x,y,y_hat):#更新权重 注意数据格式
    x=x.reshape(1,4)
    y=y.reshape(3,1)
    y_hat=y_hat.reshape(3,1)
    error_term=(y-y_hat)*x
    return error_term

def main():
    #数据准备阶段
    data=standard_data(data_test)
    feature, label = deal_data(data)
    train_feature, train_target, test_feature, test_target=data_test_train(feature, label)
    train_target=one_hot(train_target)
    test_target=one_hot(test_target)

    #设置初始权重
    n1, m1 = train_feature.shape
    n2, m2 = train_target.shape
    weight = np.random.normal(scale=1 / n1 ** 0.5, size=(m2, m1))

    #训练
    epochs=1000
    for e in range(epochs):
        loss=0
        for x,y in zip(train_feature.values,train_target.values):
            y_hat=softmax(np.dot(weight,x))#dot 多维乘一维  自动匹配行列
            weight=weight+error(x,y,y_hat)*0.2
            loss=loss+loss_derivative(y,y_hat)

        if e%100==0:
            if e!=0:
                print(e,'=========')
                last_loss=loss
                loss=loss/len(test_feature)#所有数据的的交叉熵求平均值
                if last_loss>loss :
                    print('loss=',loss)
                else:
                    print('交叉熵增加')

    #测试
    out=softmax(np.dot(weight,test_feature.values.T))
    out=out.T>0.5
    accuary=np.mean(out==test_target.values)
    print(accuary)

main()