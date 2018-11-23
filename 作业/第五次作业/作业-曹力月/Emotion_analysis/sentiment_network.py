import time
import sys
import numpy as np
from collections import Counter

"""神经网络类：含一个输入层，一个隐藏层和一个输出层"""
class SentimentNetwork:
    def __init__(self, reviews, labels, hidden_size=10, learning_rate=0.1, min_count=10,polarity_cutoff=0.1):
        # 设置随机数种子
        np.random.seed(1)

        # 数据预处理
        self.pre_process_data(reviews, labels, polarity_cutoff, min_count)
        # 初始化网络
        self.init_network(len(self.review_vocab), hidden_size, 1, learning_rate)

    # ------------------------------------------------------------------------------------------------------------------

    # 数据预处理
    def pre_process_data(self, reviews, labels, polarity_cutoff, min_count):
        """
           有策略地删减词汇，进行降噪
        """
        positive_counts = Counter()
        negative_counts = Counter()
        total_counts = Counter()

        for i in range(len(reviews)):
            if labels[i] == 'POSITIVE':
                for word in reviews[i].split(" "):
                    positive_counts[word] += 1
                    total_counts[word] += 1
            else:
                for word in reviews[i].split(" "):
                    negative_counts[word] += 1
                    total_counts[word] += 1

        pos_neg_ratios = Counter()

        for term, cnt in list(total_counts.most_common()):
            if cnt >= 50:
                pos_neg_ratio = positive_counts[term] / float(negative_counts[term] + 1)
                pos_neg_ratios[term] = pos_neg_ratio

        for word, ratio in pos_neg_ratios.most_common():
            if ratio > 1:
                pos_neg_ratios[word] = np.log(ratio)
            else:
                pos_neg_ratios[word] = -np.log((1 / (ratio + 0.01)))
        """降噪完毕"""


        # 建立评论中所有单词的词汇表
        review_vocab = set()
        for review in reviews:
            for word in review.split(" "):
                """
                降噪处理——
                    只添加最少出现min_count次数的单词，对于pos/neg比率的单词，只添加符合polarity_cutoff的单词
              """
                if total_counts[word] > min_count:
                    if word in pos_neg_ratios.keys():
                        if (pos_neg_ratios[word] >= polarity_cutoff) or (pos_neg_ratios[word] <= -polarity_cutoff):
                            review_vocab.add(word)
                    else:
                        review_vocab.add(word)
                """降噪完毕"""

        # 将评论词汇表由set格式转化为list，以便用索引获取单词，并保存在神经网络的实例对象中
        self.review_vocab = list(review_vocab)

        # 建立标签的词汇集合
        label_vocab = set()
        for label in labels:
            label_vocab.add(label)

        # 将标签词汇表由set格式转化为list，并保存在神经网络的实例对象中
        self.label_vocab = list(label_vocab)

        # 存储评论和标签词汇表的大小
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)

        # 建立评论词汇表的字典，每个单词(键)对应一个索引(值)
        self.word2index = {}
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i

        # 建立标签表的字典
        self.label2index = {}
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i

    # ------------------------------------------------------------------------------------------------------------------

    # 常规神经网络模型的初始化
    def init_network(self, input_size, hidden_size, output_size, learning_rate):
        # 初始化各层神经元的个数
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 初始化学习率
        self.learning_rate = learning_rate

        # 初始化权重——
        #           输入层到隐藏层的权重初始化为对应形状大小的默认元素为0的numpy数组
        #           隐藏层到输出层的权重初始化为对应形状的均值为0，标准差为1/(output_size**0.5)的正态分布值
        self.weights_input_hidden = np.zeros((self.input_size,self.hidden_size))
        self.weights_hidden_output = np.random.normal(0.0, self.output_size**-0.5,
                                              (self.hidden_size,self.output_size))

        """提高网络效率
        # 创建输入层，其形状为(1*input_size)大小的二维矩阵【注意是双括号】，默认元素为0
        self.layer_0 = np.zeros((1, input_size))
       """
        self.layer_hidden = np.zeros((1, hidden_size))

    """提高网络效率
    # 更新输入层数据
    def update_input_layer(self, review):
        # 清除该层之前的状态:
        self.layer_0 *= 0

        for word in review.split(" "):   # 对于单条评论
            if word in self.word2index.keys():
            
               # 消除神经网络的噪声：
               #     如果对于单词进行计数操作，会发现许多无用的单词如空格，句号等的计数较大，对加权和产生了较高的影响，这不利于神经网络的训练；
               #     这里不再统计各单词出现的次数，而只记录单词是否出现过：出现计为1，没有出现为0；
             
                self.layer_0[0][self.word2index[word]] = 1
    """

    # 更新标签数据
    def get_target_for_label(self, label):
        if(label == 'POSITIVE'):
            return 1
        else:
            return 0

    # ------------------------------------------------------------------------------------------------------------------

    # sigmoid激活函数
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # sigmoid激活函数的梯度
    def sigmoid_output_2_derivative(self, output):
        return output * (1 - output)

    # 训练神经网络——使用训练数据集
    def train(self, train_reviews_raw, train_labels):

        """
           ## 提高网络的效率：
                  处理训练评论集数据，使我们可以直接通过索引操作非0的输入
        """
        train_reviews = list()  # 列表
        for review in train_reviews_raw:
            indices = set()  # 集合
            for word in review.split(" "):
                if word in self.word2index.keys():
                    indices.add(self.word2index[word])   # 添加非0词索引
            train_reviews.append(list(indices))


        # 确定训练输入数据的个数和训练标签的个数相同
        assert(len(train_reviews_raw)) == len(train_labels)

        # 跟踪训练过程中精度的变化
        accuracy_so_far = 0

        # 记录开始打印时间统计的时刻
        start = time.time()

        ## 对于每条输入数据，进行前向传递和反向传播过程，并用梯度下降法更新权重
        for i in range(len(train_reviews_raw)):   # i是索引值
            # 得到下一条评论和它对应的标签
            review = train_reviews_raw[i]
            label = train_labels[i]

            """Forward"""

            """提高网络效率
            # 输入层——数据更新
            self.update_input_layer(review)
           """
            # 隐藏层输出，同时为输出层输入  【注意这里的输出不加激活函数】
            """提高网络效率
            layer_hidden_outputs = np.dot(self.layer_0, self.weights_input_hidden)
           """
            """仅添加对应非0值的权重"""
            self.layer_hidden *= 0
            for index in review:
                self.layer_hidden += self.weights_input_hidden[index]

            # 输出层输出   【使用sigmoid函数激活】
            layer_output_outputs = self.sigmoid(np.dot(self.layer_hidden, self.weights_hidden_output))

            """Backward"""
            # 输出层误差——预测值和标签之间的差值
            layer_output_error = layer_output_outputs - self.get_target_for_label(label)
            # 输出层误差项——输出层误差乘以激活函数的导数【非矩阵相乘，而是对应元素相乘】
            layer_output_error_delta = layer_output_error * self.sigmoid_output_2_derivative(layer_output_outputs)

            # 隐藏层误差——传递回来的输出层误差项乘以影响的权重(隐藏层到输出层)【注意：因为是反向，所以权重矩阵要进行转置】
            layer_hidden_error = np.dot(layer_output_error_delta, self.weights_hidden_output.T)
            # 隐藏层误差项——隐藏层误差乘以该层激活函数的导数[因为该层没有激活函数，所以仍等于隐藏层误差本身]
            layer_hidden_error_delta = layer_hidden_error

            # 更新各层权重  【保证形状大小和一开始的权重形状相同，这里所有的输入都进行了转置】
            self.weights_hidden_output -= self.learning_rate * np.dot(self.layer_hidden.T, layer_output_error_delta)
            """提高网络效率——只更新用于前向传播的各权重
            self.weights_input_hidden -= self.learning_rate * np.dot(self.layer_0.T, layer_output_error_delta)
           """
            for index in review:
                self.weights_input_hidden[index] -= self.learning_rate * layer_hidden_error_delta[0]


            """记录预测的精确度——目前为止预测正确的个数"""
            if layer_output_outputs >= 0.5 and label == 'POSITIVE':
                accuracy_so_far += 1
            elif layer_output_outputs < 0.5 and label == 'NEGATIVE':
                accuracy_so_far += 1

            """对于一些信息进行输出"""
            elapsed_time = float(time.time() - start)    # 记录训练的时间
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0  # 记录每秒读取的评论个数

            sys.stdout.write("\rProgress:" + str(100 * i/float(len(train_reviews_raw)))[:4]
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5]
                             + " #Accuracy:" + str(accuracy_so_far) + " #Trained:" + str(i+1)
                             + " Training Accuracy:" + str(accuracy_so_far * 100 / float(i+1))[:4] + "%")
            if i % 2500 == 0:
                print("")

    # 用神经网络进行预测
    def predict(self, review):
        """提高网络效率
        # 输入层
        self.update_input_layer(review.lower())   # 将评论中的字母都变为小写
        """
        """提高网络效率：采用self的layer_hidden而非局部的layer_hidden对象"""
        self.layer_hidden *= 0

        """提高网络效率——
                 对评论数据进行预处理，以便利用单词索引；
                 然后仅给在评论中出现过的索引增加权重，从而对隐藏层进行更新
       """
        unique_indices = set()
        for word in review.lower().split(" "):
            if word in self.word2index.keys():
                unique_indices.add(self.word2index[word])
        for index in unique_indices:
            self.layer_hidden += self.weights_input_hidden[index]

        """提高网络效率
        # 隐藏层输出
        layer_hidden_outputs = np.dot(self.layer_0, self.weights_input_hidden)
       """

        # 输出层输出
        layer_output_outputs = self.sigmoid(np.dot(self.layer_hidden, self.weights_hidden_output))

        # 将输出的数值转化为文本标签
        if layer_output_outputs[0] >= 0.5:
            return "POSITIVE"
        else:
            return "NEGATIVE"

    # 测试神经网络——使用测试数据集
    def test(self,test_reviews, test_labels):
        # 记录正确预测的个数
        correct_count = 0
        # 记录每秒做了多少预测
        start = time.time()   # 开始时间

        for i in range(len(test_reviews)):
            pred = self.predict(test_reviews[i])  # 得到预测值
            if(pred == test_labels[i]):   # 预测值和标签相等
                correct_count += 1

            # 输出信息
            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i / float(len(test_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_count) + " #Tested:" + str(i + 1) \
                             + " Testing Accuracy:" + str(correct_count * 100 / float(i + 1))[:4] + "%")


