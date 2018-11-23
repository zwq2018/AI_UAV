from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.utils import np_utils
import tensorflow as tf
import numpy as np
tf.python.control_flow_ops = tf

"""
   说明1：序列模型
          keras.models.Sequential 类是神经网络模型的封装容器；
          其提供一些常见的函数，如fit()、evaluate()、compile()
"""
# 创建一个序列模型
model = Sequential()

"""
   说明2：层
          Keras 层就像神经网络层。有全连接层、最大池化层和激活层；
          (1) 可以使用模型的add()函数添加层。
          (2) Keras会根据第一层自动推断后续所有层的形状，即只需要为第一层设置输入维度
"""
# 第一层 - 添加有128个节点的全连接层以及32个节点的输入层
model.add(Dense(128, input_dim=32))

# 第二层 - 添加softmax激活层
model.add(Activation('softmax'))

# 第三层 - 添加全连接层
model.add(Dense(10))

# 第四层 - 添加sigmoid激活层
model.add(Activation('sigmoid'))

"""
   说明3：构建好模型后，需要对其进行编译；
          在模型的compile函数中指定损失函数，优化程序和评估模型用到的指标。
"""
# 编译
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics = ['accuracy'])
# 查看模型架构
model.summary()

"""
   说明4：编译模型后，需要对模型进行拟合；
          使用fit函数训练模型，指定epoch(即训练轮数[周期])——每epoch完成对整个数据集的一次遍历；
                 keras1中为nb_epoch，Keras2中为epochs
          通过verbose参数指定显示训练过程信息类型，定义为0表示不显示信息。
"""
# 训练
# model.fit(X, y, nb_epoch=1000, verbose=0)

# 评估模型
model.evaluate()

# ----------------------------------------------------------------------------------------------------------------------

"""一般流程：加载数据；定义网络；训练网络"""
"""利用Keras构建一个简单的多层前向反馈神经网络以解决XOR问题"""

# 设置随机数种子
np.random.seed(42)

# 设置输入和标签
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).astype('float32')  # 4*2矩阵
y = np.array([[0], [1], [1], [0]]).astype('float32')   # 4*1矩阵

# 对标签进行独热编码
y = np_utils.to_categorical(y)

# 构建模型和层
xor = Sequential()
xor.add(Dense(32, input_dim=2))  # 隐层神经元个数可以调整(如64)
xor.add(Activation("relu"))
xor.add(Dense(2))
xor.add(Activation("softmax"))

# 编译
xor.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# 查看模型架构
# xor.summary()

# 训练模型
history = xor.fit(X, y, nb_epoch=1000, verbose=0)

# 评估模型
score = xor.evaluate(X, y)
print("\nAccuracy: ", score[-1])

# 获取预测值
print("\nPredictions:")
print(xor.predict_proba(X))