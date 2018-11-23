"""用 TensorFlow 来构建一个分类器来对 MNIST 数字进行分类"""
"""更关注多层神经网络的架构，而不是调参"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

# 设定超参数
learning_rate = 0.001
training_epochs = 20
batch_size = 128  # 如果没有足够内存，可以降低 batch size
display_step = 1

n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# 隐藏层参数
n_hidden_layer = 256   # 隐藏层所含的神经元个数

# 各层的权重和偏置项——变量初始化——存储为字典格式
weights = {
    'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))
}
biases = {
    'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# 处理输入数据
"""
  说明：MNIST 数据集是由 28px * 28px 单通道图片组成；
       tf.reshape()函数把 28px * 28px 的矩阵转换成了 784px * 1px 的单行向量x
"""
x = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, n_classes])

x_flat = tf.reshape(x, [-1, n_input])

"""计算各层输出"""
# ReLU作为隐藏层激活函数
layer_1 = tf.add(tf.matmul(x_flat, weights['hidden_layer']), biases['hidden_layer'])
layer_1 = tf.nn.relu(layer_1)

# 输出层的线性激活函数
logits = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])

# 定义误差值和优化器
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# 创建初始化操作
init = tf.global_variables_initializer()

"""
   说明：TensorFlow 中的 MNIST 库提供了分批接收数据的能力——
            调用mnist.train.next_batch()函数返回训练数据的一个子集。
            
         mnist.train.num_examples为训练样本的个数
"""
# 启动图
with tf.Session() as sess:
    sess.run(init)
    # 训练循环
    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples/batch_size)

        # 遍历所有 batch
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            # 运行优化器进行反向传播、计算 cost（获取 loss 值）
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
