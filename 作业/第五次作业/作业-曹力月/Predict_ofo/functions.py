# 该文件存放得到神经网络模型以及对模型进行训练时所需的函数
import numpy as np
import pandas as pd

# sigmoid激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# softmax激活函数
def softmax(x):
    x = x - np.max(x)   # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))

# ReLU激活函数
def relu(x):
    if x > 0:
        return x
    else:
        return 0

# 输出层激活函数
def activation_function(x):
    return x


# 均方误差形式的损失函数——对于单个数据
def mean_square_error(output, label):
    return np.mean((label - output)**2)


# 交叉熵误差形式的损失函数
def cross_entropy_error(output, label):
    delta = 1e-7  # 设定微小值，防止log中的参数为0

    train_size = output.shape[0]   # 得到训练集的数据个数

    return -np.sum(label * np.log(output + delta)) / train_size    # 对于所有数据求交叉熵误差


# 反向误差传递中的误差项
def error_term_formula(data, label, output):
    return -data * (label - output)


# 计算识别精度
def accuracy(weights, data, label):
    output = softmax(np.dot(data, weights))  # 得到激活后的输出值

    output = np.argmax(output, axis=1)   # 得到输出最大值的索引，概率最大的作为预测结果
    label = np.argmax(label, axis=1)   # 得到正确解标签

    # 统计所有数据中预测正确的总个数，对输入数据的个数求平均
    accuracy = np.sum(output == label) / float(data.shape[0])

    return accuracy

