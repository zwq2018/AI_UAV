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
    (train_feature, valid_feature) = train_x[5000:], train_x[:5000]
    (train_label, valid_label) = train_y[5000:], train_y[:5000]

    return train_feature, train_label, valid_feature, valid_label, test_feature, test_label, num_classes

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint

# 设计模型
def model_design():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu',
                     input_shape=(32, 32, 3)))  # 16个2*2卷积核，卷积步长为2，卷积输出的维度不变
    model.add(MaxPooling2D(pool_size=2))        # 采用最大池化，池化窗口大小为2*2
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu')) # 32个2*2卷积核
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu')) # 64个2*2卷积核
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))

    #model.summary()
    return model

# 读取数据集
(train_x, train_y), (test_feature, test_label) = cifar10.load_data() # 返回的数据类型为元组
train_y = train_y.flatten() # 降维
test_label = test_label.flatten()

# 数据预处理
train_feature, train_label, valid_feature, valid_label, test_feature, test_label, num_classes = \
    DataPreprocessing(train_x, train_y, test_feature, test_label)

# 设计模型
model = model_design()

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# 训练模型
checkpointer = ModelCheckpoint(filepath=r'E:\USTC\AI+UAV\作业\第六周\第六周作业—汪磊\CNN_classification.weights.best.hdf5',
                               verbose=1, save_best_only=True)
hist = model.fit(train_feature, train_label, batch_size=32, epochs=100,
          validation_data=(valid_feature, valid_label), callbacks=[checkpointer],
          verbose=0, shuffle=True)

# 测试模型
model.load_weights(r'E:\USTC\AI+UAV\作业\第六周\第六周作业—汪磊\CNN_classification.weights.best.hdf5')
score = model.evaluate(test_feature, test_label, verbose=0)
print('\nModel accuracy:', score[1])