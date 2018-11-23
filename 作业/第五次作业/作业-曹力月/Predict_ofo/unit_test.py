import unittest
import numpy as np
import pandas as pd
from functions import *
from two_layer_network import *

"""单元测试类：检测网络实现是否正确"""

# 初始值的设定
inputs = np.array([[0.5, -0.2, 0.1]])
targets = np.array([[0.4]])
test_w_i_h = np.array([[0.1, -0.2],
                       [0.4, 0.5],
                       [-0.3, 0.2]])
test_w_h_o = np.array([[0.3],
                       [-0.1]])

class TestMethods(unittest.TestCase):
    """
    # 测试数据集的加载
    def test_data_path(self):
        # 测试文件路径是否正确
        self.assertTrue(data_path.lower() == 'bike-sharing-dataset/hour.csv')

    def test_data_loaded(self):
        # 测试加载数据是否为DataFrame格式
        self.assertTrue(isinstance(rides, pd.DataFrame))
    """

    # 测试神经网络的功能
    """
    # 测试激活函数是否为sigmoid
    def test_activation(self):
        network = TwoLayerNetwork(3, 2, 1, 0.5)
        self.assertTrue(np.all(network.activation_function(0.5 == 1/(1+np.exp(-0.5)))))
    """

    # 测试训练神经网络时是否正确更新了权重
    def test_train(self):
        print("inputs=", inputs.shape)
        print("targets=", targets.shape)
        network = TwoLayerNetwork(3, 2, 1, 0.5)

        # 先设定权重的初始值
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        network.train(inputs, targets)

        self.assertTrue(np.allclose(network.weights_hidden_to_output,
                                    np.array([[0.37275328],
                                              [-0.03172939]])))
        self.assertTrue(np.allclose(network.weights_input_to_hidden,
                                    np.array([[0.10562014, -0.20185996],
                                              [0.39775194, 0.50074398],
                                              [-0.29887597, 0.19962801]])))

    # 测试预测函数是否正确
    def test_predict(self):
        network = TwoLayerNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        self.assertTrue(np.allclose(network.predict(inputs), 0.09998924))


suite = unittest.TestLoader().loadTestsFromModule(TestMethods())
unittest.TextTestRunner().run(suite)


