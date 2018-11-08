"""
   功能——实现2层神经网络(隐藏层为1层)的类，其将权重参数(w和b)作为属性，赋给初始化的值；
   后续在其他文件中用梯度下降法找到最优的权重参数值更新该神经网络类的属性；传入的数据集可以进行选择；
   变量说明——params：保存神经网络的参数的字典型变量。
               params['W1']是第1层的权重(与第2层的输入数组形状相同)，params['b1']是第1层的偏置。
               params['W2']是第2层的权重(与第2层的输入数组即第1层的输出数组形状相同)，params['b2']是第2层的偏置。

               grads：保存梯度的字典型变量(numerical_gradient()方法的返回值)
               grads['W1']是第1层权重的梯度，grads['b1']是第1层偏置的梯度。
               grads['W2']是第2层权重的梯度，grads['b2']是第2层偏置的梯度。
   方法说明——__init__(self,input_size,hidden_size,output_size)：进行初始化，传入的参数为输入层的神经元数，
                                        隐藏层的神经元数和输出层的神经元数。
               predict(self,x)：进行预测，由激活函数对权重的加权和进行转换得到输出值；x是传入的数据。
               loss(self,x,t)：计算损失函数的值。参数x是传入的数据，t是正确解标签。
               accuracy(self,x,t)：计算识别精度(预测的良好程度)。
               numerical_gradient(self,x,t)：计算权重参数的梯度。
"""
import sys, os
sys.path.append(os.pardir)
import numpy as np
from functions import *

class TwoLayerNet:

    """初始化"""
    def __init__(self, input_size, hidden_size, output_size):
        weight_init_std = 0.01

        # 使用符合高斯分布的随机数初始化权重，使用0初始化偏置参数
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)  # b1的个数和第一层(隐藏层)中神经元的个数相同
        self.params['W2'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    """进行预测"""
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1   # 隐藏层的加权和
        z1 = sigmoid(a1)          # 隐藏层的输出
        a2 = np.dot(z1, W2) + b2  # 输出层的加权和
        y = softmax(a2)           # 最终输出

        return y


    """计算损失函数"""
    # x：输入数据， t：监督数据
    def loss(self, x, t):
        y = self.predict(x)   # 得到预测结果

        return cross_entropy_error(y, t)   # 使用交叉熵误差作为损失函数


    """计算识别精度"""
    def accuracy(self, x, t):
        y = self.predict(x)   # 得到预测值
        y = np.argmax(y, axis=1)   # 得到输出最大值的索引，概率最大的作为预测结果
        t = np.argmax(t, axis=1)   # 得到正确解标签

        # 统计所有数据中预测正确的总个数，对输入数据的个数求平均
        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy


    """计算损失函数对于权重参数的梯度"""
    # x：输入数据，t：监督数据
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)   # 权重参数W作为自变量x传入损失函数，得到损失函数

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads






