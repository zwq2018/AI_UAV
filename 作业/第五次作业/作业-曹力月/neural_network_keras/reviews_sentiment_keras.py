"""
   INFO：数据集——25,000条IMDB评论：
                   1. 每个评论附带一个标签：负面评论的标签为0，正面评论的标签为1；
                   2. 每个评论被编码为一系列索引，对应于评论中的单词：词按频率排序，即整数1对应于最频繁的词，依次向下；
                                                                      按惯例，整数0对应于未知词；
                   3. 通过简单地连接这些整数，将句子变成一个向量。
"""

import numpy as np
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from keras import optimizers
import matplotlib.pyplot as plt

# 设置随机数种子
np.random.seed(42)

# ----------------------------------------------------------------------------------------------------------------------

"""一、加载数据"""
"""
   说明1：IMDB数据集预先加载了Keras，即不需要手动打开或读取任何文件。
   说明2：imdb数据加载函数部分参数说明——
              num_words ：查看的单词数量；
              skip_top ：忽略的热门词汇；
"""

(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=1000,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)
print("x_train.shape=", x_train.shape)
print("x_test.shape=", x_test.shape)
print(x_train)


"""二、检查数据"""
"""数据已经过预处理，评论作为向量与评论中包含的单词一起出现"""
# 输出结果是1和0的向量，其中1表示正面评论，0是负面评论


"""三、将输入向量转化为(0,1)-向量；并将输出向量转化为One-hot编码"""
"""
   说明：Tokenizer类用于向量化文本，或将文本转换为序列(即单词在字典中的下标构成的列表，从1算起)
"""
# 处理输入
tokenizer = Tokenizer(num_words=1000)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print(x_train[0])

# 处理输出
num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train.shape)
print(y_test.shape)

# ----------------------------------------------------------------------------------------------------------------------

"""四、构建模型"""
"""
   说明：dropout层用以减少过拟合
"""

sentiment_network = Sequential()   # 网络结构是1000*512*60*2
sentiment_network.add(Dense(200, activation='sigmoid', input_dim=1000))
sentiment_network.add(Dropout(0.3))
sentiment_network.add(Dense(100, activation='relu'))
sentiment_network.add(Dropout(0.3))
sentiment_network.add(Dense(50, activation='relu'))
sentiment_network.add(Dropout(0.3))
sentiment_network.add(Dense(num_classes, activation='softmax'))   # 输出层
sentiment_network.summary()

# 编译
sentiment_network.compile(loss='categorical_crossentropy',
                          optimizer='Adadelta',
                          metrics=['accuracy'])

# 训练
history = sentiment_network.fit(x_train, y_train,
                                batch_size=500,
                                epochs=100,
                                validation_data=(x_test, y_test),
                                verbose=2)

# ----------------------------------------------------------------------------------------------------------------------

"""五、评估模型"""
score = sentiment_network.evaluate(x_test, y_test, verbose=0)
print(score)
print("Accuracy: ", score[1])