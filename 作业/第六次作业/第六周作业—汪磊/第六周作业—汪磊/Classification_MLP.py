import keras
from keras.datasets import cifar10
import numpy as np

# 数据预处理
def DataPreprocessing(train_x, train_y, test_feature, test_label):
    # 数据归一化
    train_x = train_x.astype('float32') / 255
    test_feature = test_feature.astype('float32') / 255

    # 对label进行one-hot编码
    num_classes = len(np.unique(train_y))   # np.unique去除重复元素，并按元素由大到小返回一个新的无重复的元组或者列表
    train_y = keras.utils.to_categorical(train_y, num_classes) # 对train_y进行one-hot编码，位数为num_classes
    test_label = keras.utils.to_categorical(test_label, num_classes)

    # 生成训练集和验证集
    (train_feature, valid_feature) = train_x[5000:], train_x[:5000] # 45000*32*32*3, 5000*32*32*3
    (train_label, valid_label) = train_y[5000:], train_y[:5000]     # 45000*10, 5000*10

    return train_feature, train_label, valid_feature, valid_label, test_feature, test_label, num_classes

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint

# 设计模型：输入层1000，隐藏层512，输出层num_classes(分类数量)
def model_design(num_classes):
    model = Sequential()
    model.add(Flatten(input_shape=train_feature.shape[1:]))  # feature的shape分别是样本数，通道数，图宽度，图高度，input_shape不能指定样本数
    model.add(Dense(1000, activation='relu'))  # 添加全连接层，并选择相应的激活函数
    model.add(Dropout(0.2))  # 该层神经元被dropout的概率为0.2
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    return model

# 读取数据集
(train_x, train_y), (test_feature, test_label) = cifar10.load_data() # 返回的数据类型为元组
train_y = train_y.flatten() # 降维
test_label = test_label.flatten()

# 数据预处理
train_feature, train_label, valid_feature, valid_label, test_feature, test_label, num_classes = \
    DataPreprocessing(train_x, train_y, test_feature, test_label)

# 设计模型
model = model_design(num_classes)

# 编译模型：本题中sgd比RMSprop优化效果更好
sgd = keras.optimizers.SGD(lr=0.05, decay=5e-4, momentum=0.9, nesterov=True)
    # decay为参数更新后学习率衰减值，keras内置公式LearningRate = LearningRate * 1/(1 + decay * epoch)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 训练模型
checkpointer = ModelCheckpoint(filepath=r'E:\USTC\AI+UAV\作业\第六周\第六周作业—汪磊\MLP.weights.best.hdf5',
                               verbose=1, save_best_only=True)     # 保存最佳权重参数
hist = model.fit(train_feature, train_label, batch_size=100, epochs=25,
          validation_data=(valid_feature, valid_label), callbacks=[checkpointer],
          verbose=0, shuffle=True) # 每次随机的batch为100，迭代25次
          # shuffle为布尔值时，表示是否在每一次epoch训练前随机打乱输入样本的顺序

# 测试模型
model.load_weights(r'E:\USTC\AI+UAV\作业\第六周\第六周作业—汪磊\MLP.weights.best.hdf5')
score = model.evaluate(test_feature, test_label, verbose=0)
print('\nModel accuracy:', score[1])

# 由于程序执行时间较长，参数调整的次数并不是很多，模型精确度最高约为0.52