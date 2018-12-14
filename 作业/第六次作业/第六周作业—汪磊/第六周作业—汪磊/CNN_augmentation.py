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
from keras.preprocessing.image import ImageDataGenerator

# 模型设计
def model_design():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu',
                     input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())   # 降维后输入到全连接层
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))

    model.summary()
    return model

# 读取数据集
(train_x, train_y), (test_feature, test_label) = cifar10.load_data() # 返回的数据类型为元组
train_y = train_y.flatten() # 降维
test_label = test_label.flatten()

# 数据预处理
train_feature, train_label, valid_feature, valid_label, test_feature, test_label, num_classes = \
    DataPreprocessing(train_x, train_y, test_feature, test_label)

# 数据增强：批量生成数据，防止模型过拟合，提高泛化能力（数据集较小时十分必要）
aug_generator = ImageDataGenerator(
    width_shift_range=0.1,      # 图片随机水平偏移的幅度为0.1
    height_shift_range=0.1,     # 图片随机竖直偏移的幅度为0.1
    horizontal_flip=True)       # 随机的对图片进行水平翻转（只适用于水平翻转不影响语义的图片）
aug_generator.fit(train_feature)

# 设计模型
model = model_design()

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# 训练模型
checkpointer = ModelCheckpoint(filepath=r'E:\USTC\AI+UAV\作业\第六周\第六周作业—汪磊\aug_CNN.weights.best.hdf5',
                               verbose=1, save_best_only=True)
batch_size = 100
model.fit_generator(aug_generator.flow(train_feature, train_label, batch_size=batch_size), steps_per_epoch=train_feature.shape[0] // batch_size,
                validation_data=(valid_feature, valid_label), validation_steps=valid_feature.shape[0] // batch_size,
                callbacks=[checkpointer], epochs=100, verbose=0)

# 测试模型
model.load_weights(r'E:\USTC\AI+UAV\作业\第六周\第六周作业—汪磊\aug_CNN.weights.best.hdf5')
score = model.evaluate(test_feature, test_label, verbose=0)
print('\nModel accuracy:', score[1])