import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials import mnist


def get_weight(shape):
    weights = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    return weights
def get_bias(shape):
    bias = tf.Variable(tf.constant(0.01, shape=shape))  # 将偏置系数初值设置为0.01
    return bias
def get_shape(INPUT_NODE,LAYER1_NODE,OUTPUT_NODE):
    shape1 = [INPUT_NODE,LAYER1_NODE]
    shape2 = [LAYER1_NODE]
    shape3 = [LAYER1_NODE,OUTPUT_NODE]
    shape4 = [OUTPUT_NODE]
    return shape1,shape2,shape3,shape4
class Bp_Neural_NetworK(object):
    def __init__(self,REGULARIZATION_RATE,BATCH_SIZE,INPUT_NODE,LAYER1_NODE,OUTPUT_NODE,
                 MOVING_AVERAGE_DECAY,LEARNING_RATE_BASE,LEARNING_RATE_DECAY):
        self.REGULARIZATION_RATE = REGULARIZATION_RATE
        self.MOVING_AVERAGE_DECAY = MOVING_AVERAGE_DECAY
        self.LEARNING_RATE_BASE = LEARNING_RATE_BASE
        self.BATCH_SIZE = BATCH_SIZE
        self.LEARNING_RATE_DECAY = self.MOVING_AVERAGE_DECAY
        self.INPUT_NODE = INPUT_NODE
        self.LAYER1_NODE = LAYER1_NODE
        self.OUTPUT_NODE = OUTPUT_NODE
        self.shape1,self.shape2,self.shape3,self.shape4 = get_shape(self.INPUT_NODE,self.LAYER1_NODE,self.OUTPUT_NODE)
        self.W_hin = get_weight(self.shape1)
        self.bias_hin = get_bias(self.shape2)
        self.W_hout = get_weight(self.shape3)
        self.bias_hout = get_bias(self.shape4)


    def Forward_Pass(self,features,targets):
        '''
        定义神经网络前向传播过程
        '''

        hidden_outputs_ori = tf.matmul(features,self.W_hin)+self.bias_hin  #隐含层经过线性映射后的输出
        # hidden_outputs = tf.nn.relu(hidden_outputs_ori) #隐含层经过非线性激活函数后的映射到高维空间，这里采用relu函数
        # hidden_outputs = tf.nn.sigmoid(hidden_outputs_ori)
        hidden_outputs = tf.nn.tanh(hidden_outputs_ori)
        final_outputs = tf.matmul(hidden_outputs,self.W_hout)+self.bias_hout
        return final_outputs


    def Train(self,Training_steps,features,targets,isTrain):
        '''
        定义神经网络反向传播,即训练过程
        '''
        shape1,shape2,shape3,shape4 = get_shape(self.INPUT_NODE,self.LAYER1_NODE,self.OUTPUT_NODE)
        n_inputs = np.shape(features[0])[0]#每个输入样本维度相同，任意采样获取样本维度
        # n_class = np.shape(np.unique(self.targets)) #对标签数组去重后求样本类型总数
        # n_class = np.shape(self.targets)[1]
        n_class = self.OUTPUT_NODE
        numofinputs = np.shape(features)[0]
        Input_Samples = tf.placeholder(tf.float32,[None,n_inputs])

        # Y_ = tf.placeholder(tf.float32,[None,n_class])
        Y_ = tf.placeholder(tf.float32)
        final_outputs = self.Forward_Pass(features,targets)
        global_step = tf.Variable(0, trainable = False)

        # # 对于分类问题，定义交叉熵损失
        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     logits=final_outputs, labels=tf.argmax(Y_, 1))
        # cross_entropy_mean = tf.reduce_mean(cross_entropy)
        #对于回归问题，定义均方误差损失函数
        MSE_mean = tf.reduce_mean(tf.square(tf.subtract(final_outputs,Y_)))
        # 定义L2正则化器
        regularizer = tf.contrib.layers.l2_regularizer(self.REGULARIZATION_RATE)
        regularization = regularizer(self.W_hin) + regularizer(self.W_hout)
        # loss = cross_entropy_mean + regularization  # 总损失值
        loss = MSE_mean + regularization

        variable_averages = tf.train.ExponentialMovingAverage(
                        self.MOVING_AVERAGE_DECAY, global_step) # 传入当前迭代轮数参数
        # 定义对所有可训练变量trainable_variables进行更新滑动平均值的操作op
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # 定义指数衰减学习率
        learning_rate = tf.train.exponential_decay(self.LEARNING_RATE_BASE, global_step,
                                                   numofinputs / self.BATCH_SIZE, self.LEARNING_RATE_DECAY)
        # 定义梯度下降操作op，global_step参数可实现自加1运算
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        # train_step = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)
        # 组合两个操作op
        train_op = tf.group(train_step, variables_averages_op)
        #定义评估函数
        # correct_pred = tf.equal(tf.argmax(final_outputs,1),tf.argmax(Y_,1))
        # accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=1)
        with tf.Session() as sess:
            sess.run(init)
            step = 0
            if isTrain:
                while step*self.BATCH_SIZE < Training_steps:
                    # start = (step*self.BATCH_SIZE) % numofinputs
                    start = (step * self.BATCH_SIZE)%numofinputs
                    end = start + self.BATCH_SIZE
                    batch_x,batch_y = features[start:end],targets[start:end]
                    sess.run(train_op,feed_dict={Input_Samples:batch_x,Y_:batch_y})
                    saver.save(sess, 'ckpt/task2.ckpt', global_step = step + 1)
                    if step*self.BATCH_SIZE%700 == 0:
                        loss_now = sess.run(loss,feed_dict={Input_Samples:batch_x,Y_:batch_y})
                        print('After {} steps'.format(str(step*self.BATCH_SIZE))+",loss is {}".format(loss_now))
                    step+=1
                print("Optimization Finished")
                sess.close()
            else:
                model_file = tf.train.latest_checkpoint('ckpt/')
                saver.restore(sess, model_file)
                val_loss = sess.run(loss, feed_dict={Input_Samples:features,Y_:targets})
                print('val_loss:%f' % (val_loss))




















