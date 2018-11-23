import numpy as np
import pandas as pd
from functions import *

"""
   两层神经网络(含一层隐藏层)的类文件
   INFO：隐藏层级使用S型函数作为激活函数；
         输出层只有一个节点，用于递归，使用f(x)=x作为激活函数
"""

class TwoLayerNetwork:
    # 初始化
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        # 设定各层的神经元个数
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 初始化两个连接层权重参数——使用Xavier初始值
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_size**-0.5,
                                                        (self.input_size, self.hidden_size))
        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_size**-0.5,
                                                         (self.hidden_size, self.output_size))
        # 初始化学习率
        self.lr = learning_rate

        # 设定激活函数
        self.activation_function = sigmoid

    # 训练神经网络——单次训练
    def train(self, features, targets):
        n_records = features.shape[0]   # 训练数据的个数

        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)

        # 依次对每行数据进行处理
        for x, y in zip(features, targets):
            """Forward"""
            # 隐藏层
            hidden_inputs = np.dot(x[None, :], self.weights_input_to_hidden)
            hidden_outputs = sigmoid(hidden_inputs)
            # print("hidden_outputs=", hidden_outputs.shape)

            # 输出层
            output_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
            output_outputs = output_inputs   # 输出层激活函数为f(x)=x


            """Backward"""
            # 输出层误差
            output_error = y - output_outputs
            # print("output_error=", output_error.shape)
            # 反向传播——输出层误差项
            output_error_term = output_error * 1  # 输出层激活函数的导数是1

            # 隐藏层误差
            hidden_error = np.dot(output_error_term, self.weights_hidden_to_output.T)
            # print("hidden_error=", hidden_error.shape)
            #反向传播——隐藏层误差项
            hidden_error_term = hidden_error * hidden_outputs * (1-hidden_outputs)  # 隐藏层采用S型激活函数，其导数为y*(1-y)

            # 权重步长——输入层到隐藏层
            delta_weights_i_h += np.dot(x[:, None], hidden_error_term)
            # print("delta_weights_i_h=", delta_weights_i_h.shape)
            # 权重步长——隐藏层到输出层
            delta_weights_h_o += np.dot(hidden_outputs.T, output_error_term)
            # print("delta_weights_h_o=", delta_weights_h_o.shape)


        # 所有数据训练完后对两层权重进行更新
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records

    # 获取神经网络的预测值（实质进行前向传播）
    def predict(self, features):
        # 隐藏层
        hidden_inputs = features
        hidden_outputs = sigmoid(np.dot(features, self.weights_input_to_hidden))

        # 输出层
        output_inputs = hidden_outputs
        output_outputs = activation_function(np.dot(output_inputs, self.weights_hidden_to_output))

        return output_outputs

