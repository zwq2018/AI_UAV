import tensorflow as tf

INPUT_NODE = 784  # 输入层的节点数
OUTPUT_NODE = 10  # 输出层的节点数
LAYER1_NODE = 500  # 隐藏层的节点数


# 获取权重
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', regularizer(weights))
    return weights


# 定义一个使用了Relu函数的三层全连接神经网络，同时引入参数平均值的类，方便测试时使用滑动平均模型
def inference(input_tensor, regularizer=None):
    with tf.variable_scope('layer1'):   # 隐藏层的前向传播结果
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):  # 输出层的前向传播结果
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2