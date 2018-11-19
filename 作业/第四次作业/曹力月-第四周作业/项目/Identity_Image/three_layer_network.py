"""
   该文件实现3层神经网络(隐藏层为1层)的类(模型)，其中权重参数(w和b)作为该类的属性
   - - - - - -
   通过其它文件利用梯度下降法对该神经网络模型进行训练，找到最优的权重参数值，并更新模型
"""
import numpy as np
import pandas as pd
from functions import *

class ThreeLayerNet:

    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        weight_init_std = 0.01  # 设置初始化权重时的标准差为0.01

        """
        # 使用由标准差为0.01的高斯分布产生的值初始化权重，使用0初始化偏置
        self.params = {}  # 字典变量

        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden1_size)  # W1是一个二维数组，行数为输入数据的维度，列数为第1隐藏层的神经元个数
        self.params['b1'] = np.zeros_like(hidden1_size)  # b1的个数和第1隐藏层中神经元的个数相同

        self.params['W2'] = weight_init_std * np.random.randn(hidden1_size, hidden2_size)
        self.params['b2'] = np.zeros_like(hidden2_size)

        self.params['W3'] = weight_init_std * np.random.randn(hidden2_size, output_size)
        self.params['b3'] = weight_init_std * np.random.randn(output_size)
      """
        weights = weight_init_std * np.random.randn(input_size, hidden1_size, hidden2_size, output_size)   # 用标准差为0.01的高斯分布初始化各层权重


# 由该神经网络模型得到预测值
def predict(self, data):

    """
    W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']   # 得到各层初始化后的权重参数值[分别为一个二维数组]
    b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']   # 得到各层初始化后的偏置参数值[分别为一个向量]

    a1 = np.dot(data, W1) + b1  # 得到第1隐藏层的加权和
    p1 = sigmoid(a1)    # 得到第1隐藏层的输出
    a2 = np.dot(p1, W2)
    p2 = sigmoid(a2)
    a3 = np.dot(p2, W3)
    output = softmax(a3)    # 得到最终的预测值
    """
    output = softmax(np.dot(data, self.weights))
    return output


# 得到该神经网络的预测值，和训练数据中的标签计算损失函数
def loss(self, data, label):
    output = self.predict(data)

    return cross_entropy_error(output, label)   # 使用交叉熵误差函数


# 计算基于误差反向传播算法的误差项
def error_term_bp(self, output, label):
    return (label-output) * output * (1 - output)


# 使用梯度下降法更新参数
def bp_update_params(self, data, label, learnrate):
    # 计算误差项
    output = self.predict(data)
    error_term = error_term_bp(self, output, label)

    # 得到最终的偏导结果
    del_w = np.zeros(self.weights.shape)
    del_w += error_term * data

    # 更新权重参数
    self.weights += learnrate * del_w
