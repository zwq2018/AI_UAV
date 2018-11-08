"""
   该部分保存训练神经网络所需要的函数——
         包括sigmoid激活函数，sigmoid函数的导数，以及二分类的交叉熵误差计算
"""
import numpy as np


# sigmoid激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid的导数式
def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))

# 二分类的交叉熵误差
def error_formula(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)