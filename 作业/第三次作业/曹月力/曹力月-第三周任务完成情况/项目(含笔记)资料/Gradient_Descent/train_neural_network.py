"""
   功能——1. 该程序对TwoLayerNetwork进行训练，实现方式为mini-batch学习：
                  即完成从训练数据中随机选择一部分数据(mini-batch)，以这些mini-batch为对象，
                  使用梯度下降法更新权重和偏置参数的过程。
           2. 基于测试数据对训练结果进行评价，观察神经网络模型的泛化能力，防止过拟合。
                  定期对训练数据和测试数据记录识别精度，每隔一个epoch记录一次[epoch=所有训练数据均被使用过一次时的更新次数]
   数据集——使用MNIST数据集进行训练。
   更新参数的方法——随机梯度下降法(SGD)，由于数据集是随机选取的mini-batch数据。
"""

import numpy as np
from two_layer_network import TwoLayerNet
from mnist_access import load_mnist   #  现成的加载MNIST数据集的方法

# 获取训练数据集和测试数据集
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label= True)

#  超参数[从数据集中获得或需要人工指定的参数]
iters_num = 10000   # 设定更新(迭代)次数
train_size = x_train.shape[0]   # 训练数据的个数
batch_size = 100    # 设定小批量的数据大小为100个
learning_rate = 0.1    # 设定学习率为0.1(通用设定)

# 定义记录学习过程的各个函数列表
train_loss_list = []  # 保存每次更新后损失函数的值
train_acc_list = []   # 保存每次更新后对训练数据的识别精度的值
test_acc_list = []    # 保存每次更新后对测试数据的识别精度的值
# epoch的大小，即更新多少次之后记录一次识别精度
iter_per_epoch = max(train_size / batch_size, 1)

# 初始化神经网络
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # 每次更新后都要获取mini-batch
    batch_mask = np.random.choice(train_size, batch_size)  # 随机从train_size个数中获取batch_size个索引值[即训练数据集中数据的索引]
    x_batch = x_train[batch_mask]   # 获取mini-batch的全部输入
    t_batch = t_train[batch_mask]   # 获取mini-batch的正确解集合

    # 基于数值微分的方法计算梯度
    grad = network.numerical_gradient(x_batch, t_batch)

    # 用随机梯度下降法(SGD)对每个参数进行更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 记录学习过程
    # 即在每次更新后求对应的损失函数，每个epoch求训练精度和测试精度
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    print("loss = ")
    print( loss)

    if i % iter_per_epoch == 0:   # 到达了一个新纪元
        train_acc = network.accuracy(x_train, t_train)
        train_acc_list.append(train_acc)
        test_acc = network.accuracy(x_test, t_test)
        test_acc_list.append(test_acc)
        print("train acc, test acc |" + str(train_acc) + ',' + str(test_acc))



