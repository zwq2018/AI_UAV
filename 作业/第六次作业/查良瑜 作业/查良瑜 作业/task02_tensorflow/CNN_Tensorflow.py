import tensorflow as tf
import numpy as np
class CNN(object):#
    def __init__(self,learning_rate,batch_size,REGULARIZATIONRATE):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.REGULARIZATIONRATE = REGULARIZATIONRATE  #正则化率
        #W2 = (W1-F+2P)/S + 1 ，W1为上一层的宽度，F为感受野宽度，P是zero-padding宽度，S为步幅
        #H2 = (H1-F+2P)/S + 1,H1为上一层的高度
        self.Weights = {
            "wc1":tf.Variable(tf.random_normal([3,3,1,32])),   #第一层卷积核大小为3*3，将输入图片映射到32通道的featuremap上
            "wc2":tf.Variable(tf.random_normal([3,3,32,64])),  #第二层卷积核大小为3*3，将上一层32通道featuremap映射到下一层64通道features
            "wc3":tf.Variable(tf.random_normal([3,3,64,96])),
            "wc4":tf.Variable(tf.random_normal([3,3,96,96])),
            'wd1':tf.Variable(tf.random_normal([4*4*96,2*4*4*96+1])),#隐含层节点个数为S=2*n+1,n为上一层参数个数
            "wd2":tf.Variable(tf.random_normal([3073,3073*2+1])),
            'out':tf.Variable(tf.random_normal([6147,10]))
        }
        self.Biases = {
            "bc1":tf.Variable(tf.random_normal([32])),
            'bc2':tf.Variable(tf.random_normal([64])),
            'bc3':tf.Variable(tf.random_normal([96])),
            "bc4":tf.Variable(tf.random_normal([96])),
            'bd1':tf.Variable(tf.random_normal([3073])),
            'bd2':tf.Variable(tf.random_normal([6147])),
            'out':tf.Variable(tf.random_normal([10])) #10为最终样本类别
        }
    #使用tf.train产生label_batch和image_batch
    def get_batch_data(self,images,labels):
        images = tf.cast(images, tf.float32)
        labels = tf.cast(labels,tf.float32)
        input_queue = tf.train.slice_input_producer([images, labels], shuffle=False)
        image_batch, label_batch = tf.train.batch(input_queue, batch_size=self.batch_size, num_threads=1, capacity=128)
        return image_batch, label_batch


    #定义alex_net的前向传播过程
    def model_foward(self,images,labels,conv_drop,hidden_drop):#池化层dropout为0.9，全连接层dropout为0.5
        x = tf.reshape(images,[-1,28,28,1])
        #第一层卷积操作,卷积核大小为3*3，将输入图片映射到32通道的featuremap中,每个featuremap尺寸为28*28*32
        conv1 = tf.nn.conv2d(input=x,filter=self.Weights['wc1'],strides=[1,1,1,1],padding='SAME')
        conv1 = tf.nn.bias_add(conv1,bias=self.Biases['bc1'])
        conv1 = tf.nn.relu(conv1)
        #第一层最大池化下采样,池化尺寸2*2，X,Y方向移动步长为2，生成的featuremap宽高为（28-2+0）/2+1 =14，尺寸为14*14*32
        maxpool1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        #第一层规范化操作
        norm1 = tf.nn.lrn(maxpool1,depth_radius=4.0,bias=1,alpha=0.01/9.0,beta=0.75)
        #第一层池化
        drop1 = tf.nn.dropout(norm1,conv_drop)

        #第二层卷积操作，卷积核大小为3*3，将14*14*32featuremap映射到64通道的featuremap中，该层featuremap尺寸为14*14*64
        conv2 = tf.nn.conv2d(input=drop1,filter=self.Weights['wc2'],strides=[1,1,1,1],padding='SAME')
        conv2 = tf.nn.bias_add(conv2,bias=self.Biases['bc2'])
        conv2 = tf.nn.relu(conv2)
        #第一层最大池化下采样，池化尺寸2*2，X，Y方向移动步长2，将14*14*64featuremap下采样到宽高为（14-2+0）/2+1=7,尺寸为7*7*64的featuremap中
        maxpool2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        #第一层规范化操作
        norm2 = tf.nn.lrn(maxpool2,depth_radius=4.0,bias=1,alpha=0.01/9.0,beta=0.75)
        #第一层池化
        drop2 = tf.nn.dropout(norm2,conv_drop)

        #第三层卷积操作,卷积核大小为3*3，将7*7*64映射到96通道的featuremap中,每个featuremap尺寸为7*7*96
        conv3 = tf.nn.conv2d(input=drop2,filter=self.Weights['wc3'],strides=[1,1,1,1],padding='SAME')
        conv3 = tf.nn.bias_add(conv3,bias=self.Biases['bc3'])
        conv3 = tf.nn.relu(conv3)
        #第三层最大池化下采样,池化尺寸2*2，X,Y方向移动步长为2，生成的featuremap宽高为（7-2+1）/2+1 =4，尺寸为4*4*96
        maxpool3 = tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        #第一层规范化操作
        norm3 = tf.nn.lrn(maxpool3,depth_radius=4.0,bias=1,alpha=0.01/9.0,beta=0.75)
        #第一层池化
        drop3 = tf.nn.dropout(norm3,conv_drop)

        #第四层卷积操作,featuresmap尺寸最终变为4*4*96
        conv4 = tf.nn.conv2d(input=drop3,filter=self.Weights['wc4'],strides=[1,1,1,1],padding='SAME')
        conv4 = tf.nn.bias_add(conv4,bias=self.Biases['bc4'])
        conv4 = tf.nn.relu(conv4)
        #本层不设下采样层，直接进行规范化操作
        norm4 = tf.nn.lrn(conv4,depth_radius=4.0,bias=1,alpha=0.01/9.0,beta=0.75)
        #第一层池化
        drop4 = tf.nn.dropout(norm4,conv_drop)
        conv_out = tf.reshape(drop4,[-1,self.Weights['wd1'].get_shape().as_list()[0]])

        #第一个全连接层
        fc1 = tf.matmul(conv_out,self.Weights["wd1"]) + self.Biases['bd1']
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1,hidden_drop)

        #第二个全连接层
        fc2 = tf.matmul(fc1,self.Weights['wd2']) + self.Biases['bd2']
        fc2 = tf.nn.relu(fc2)
        fc2 = tf.nn.dropout(fc2,hidden_drop)

        #输出层
        predict = tf.matmul(fc2,self.Weights['out']) + self.Biases['out']
        return predict

    def train(self,images,labels,cvdrop,hddrop):
        conv_keep_prob = tf.placeholder(tf.float32)
        hidden_keep_prob = tf.placeholder(tf.float32)
        X = tf.placeholder(tf.float32,shape=[None,28,28])
        Y = tf.placeholder(tf.float32,shape=[None,10])
        predict = self.model_foward(images,labels,cvdrop,hddrop)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict,labels=labels))
        # regularizer = tf.contrib.layers.l1_regularizer(self.REGULARIZATIONRATE) #这里采用L1正则化
        # for value in self.Weights.values():
        #     regularizer += regularizer(value)
        # cost_total = cost + regularizer
        train_op = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate).minimize(cost)
        correct_pred = tf.equal(tf.argmax(predict,1),tf.argmax(labels,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
        init = tf.global_variables_initializer()
        # sv = tf.train.Supervisor()
        # with sv.managed_session() as sess:
        with tf.Session() as sess:
            batch_image,batch_label = self.get_batch_data(images,labels)
            sess.run(init) #初始化全局参数
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess,coord)
            batch_y = sess.run(batch_image)
            batch_x = sess.run(batch_label)
            step = 0
            epoch = 1
            # idx = 0
            # start = 0
            while epoch<=10:
                for step in range(100000):
                    # if idx + self.batch_size<len(labels):
                    #     idx = 0
                    # else:
                    #     start = idx
                    #     idx+=self.batch_size
                    # batch_x = images[start:start+self.batch_size]
                    # batch_y = labels[start:start+self.batch_size]
                    sess.run(train_op,feed_dict={X:batch_x,Y:batch_y,conv_keep_prob:cvdrop,hidden_keep_prob:hddrop})
                    if step%10000 == 0:
                        loss,acc = sess.run([cost,accuracy],feed_dict={X:batch_x,Y:batch_y,
                                                                             conv_keep_prob:cvdrop,
                                                                             hidden_keep_prob:hddrop
                        })
                        print('after {} epochs'.format(epoch)+",{} steps".format(step)+',loss now is {}'.format(loss)
                              +",accuracy now is {]".format(acc))
                epoch+=1

























