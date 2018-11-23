import tensorflow as tf
import notmnist_inference  # 加载 notmnist_inference.py中定义的常量和前向传播的函数
import os
import pickle  # 它可以将对象转换为一种可以传输或存储的格式 序列化过程将文本信息转变为二进制数据流
import math
import tqdm
# import notmnist_data

# 配置神经网络的参数
BATCH_SIZE = 128  # 一次训练batch中的训练数据个数。数字越小时，越接近随机梯度下降否则是梯度下降
LEARNING_RATE_BASE = 0.8  # 基础的学习速率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001  # 描述模型复杂度的正则化项在损失函数的系数
TRAINING_STEPS = 10  # 训练的轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率

MODEL_SAVE_PATH = "notMNIST_model/"
MODEL_NAME = "notmnist_model"


# 训练过程
def train(train_features, train_labels):
    # 定义输入输出的格式
    x = tf.placeholder(tf.float32, [None, notmnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, notmnist_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)  # L2正则化损失函数
    y = notmnist_inference.inference(x, regularizer)  # 前向传播求预测值y
    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # 在所有代表神经网络参数的变量上使用滑动平均
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算交叉熵并取平均值，求最终的损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 指数衰减学习速率的设置
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,  # 基础学习速率
        global_step,  # 当前迭代的轮数
        len(train_features) / BATCH_SIZE,  # 总的需要迭代的次数
        LEARNING_RATE_DECAY,  # 学习速率衰减的速度
        staircase=True)

    # 使用tf.train.GradientDescentOptimizer优化算法来优化损失函数，包含交叉熵和L2正则化损失
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 更新神经网络的参数和更新每一个参数的滑动平均值
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 保存当前的模型
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()  # 初始化会话并运行
        batch_count = int(math.ceil(len(train_features) / BATCH_SIZE))  # 将整个样本分割成142500/128个子集
        for steps_i in range(TRAINING_STEPS):
            # The training cycle
            for batch_i in range(batch_count):  # 针对每个子集
                # Get a batch of training features and labels
                batch_start = batch_i * BATCH_SIZE
                batch_features = train_features[batch_start:batch_start + BATCH_SIZE]
                batch_labels = train_labels[batch_start:batch_start + BATCH_SIZE]

                _, loss_value, step = sess.run([train_op, loss, global_step],
                                               feed_dict={x: batch_features, y_: batch_labels})

                if step % 1000 == 0:
                    print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    pickle_file = r'D:\中科大软院\数据集\notMNIST.pickle'
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)
        train_features = pickle_data['train_dataset']
        train_labels = pickle_data['train_labels']
        del pickle_data  # Free up memory
    train(train_features, train_labels)


if __name__ == '__main__':
    tf.app.run()


