"""notMNIST数据集包含不同字体的字母A-J的图片；共有500,000张"""

import hashlib
import os
import pickle
import math
from urllib.request import urlretrieve

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import  resample
from tqdm import tqdm
from zipfile import ZipFile

import tensorflow as tf
import matplotlib.pyplot as plt

"""
  :goal  对比网络模型的预测值和正确标签，通过调整模型参数使网络模型至少达到80%的预测精度 
  :steps 1.读取数据集，通过数组的形式读取进来
         2.归一化数据，读进来的是灰度图，需要将其归一化至0-1区间，便于计算梯度
         3.采用tensorflow来构建神经网络模型
         4.训练神经网络，调整超参数，提高精度
"""

"""对每组类别(A-J)，使用15,000张图片"""
"""从提供的url网址中下载文件"""
"""
   说明：urllib模块提供的urlretrieve()方法直接将远程数据下载到本地；
   
   # urlretrieve(url, filename=None, reporthook=None, data=None)
   #  1. 参数filename指定了保存本地路径（如果参数未指定，urllib会生成一个临时文件保存数据。）
   #  2. 参数reporthook是一个回调函数，当连接上服务器、以及相应的数据块传输完毕时会触发该回调，我们可以利用这个回调函数来显示当前的下载进度。
   #  3. 参数data指post导服务器的数据，该方法返回一个包含两个元素的(filename, headers) 元组，filename 表示保存到本地的路径，header表示服务器的响应头
"""


def download(url, file):
    if not os.path.isfile(file):
        print('Downloading ' + file + '...')
        urlretrieve(url, file)
        print('Download Finished')


"""下载训练集和测试集压缩包"""
# download('https://s3.amazonaws.com/udacity-sdc/notMNIST_train.zip','notMNIST_train.zip')
# download('https://s3.amazonaws.com/udacity-sdc/notMNIST_test.zip', 'notMNIST_test.zip')

"""确保文件没有被篡改"""
"""
   说明：Python的hashlib提供了常见的摘要算法，如MD5，SHA1等等；
         摘要算法又称哈希算法、散列算法。它通过一个函数(单向函数)，把任意长度的数据
         转换为一个长度固定的数据串（通常用16进制的字符串表示），这个数据串就是摘要[digest]；
         其目的是为了发现原始数据是否被人篡改过。
"""
#  ****************************************************************************
assert hashlib.md5(open('notMNIST_train.zip', 'rb').read()).hexdigest() == 'c8673b3f28f489e9cdf3a3d74e2ac8fa',\
        'notMNIST_train.zip file is corrupted.  Remove the file and try again.'
assert hashlib.md5(open('notMNIST_test.zip', 'rb').read()).hexdigest() == '5d3c7e653e63471c88df796156a9dfa9',\
        'notMNIST_test.zip file is corrupted.  Remove the file and try again.'
print('All files downloaded')
#  ****************************************************************************

"""对训练集和测试集压缩包进行解压"""
"""
   说明1：tqdm是一个快速，可扩展的python进度条，可以在python长循环中添加一个
         进度提示信息，用户只需要封装任意的迭代器tqdm(iterator)
   说明2：zipfile是python中用来做zip格式编码的压缩和解压缩的，其有两个类——
          1. ZipFile：用来创建和读取zip文件；
          2. ZipInfo：存储zip文件的每个文件的信息
   说明3：PIL(python imaging library)，是python平台上的图像处理标准库
   说明4：numpy库中的flatten函数，返回一个折叠成一维的数组；只能用于numpy对象，如array或mat，而不能作用于list列表
"""
def uncompress_features_labels(file):
    features = []
    labels = []

    with ZipFile(file) as zipf:
        # 进度条
        filename_pbar = tqdm(zipf.namelist(), unit='files')

        # 从所有文件中得到特征和标签
        for filename in filename_pbar:
            # 检查文件是否为文件夹
            if not filename.endswith('/'):
                with zipf.open(filename) as image_file:
                    image = Image.open(image_file)
                    image.load()
                    # 加载图像数据为一维的数组，并使用float32类型来存储
                    feature = np.array(image, dtype=np.float32).flatten()

                # 得到图片的标签(字母)
                label  = os.path.split(filename)[1][0]

                features.append(feature)
                labels.append(label)
    return np.array(features), np.array(labels)   # 返回特征和标签的numpy数组


# 调用函数，从压缩文件中得到特征和标签
train_features, train_labels = uncompress_features_labels('notMNIST_train.zip')
test_features, test_labels = uncompress_features_labels('notMNIST_test.zip')

# 限定图片数量，随机取样
docker_size_limit = 150000
train_features, train_labels = resample(train_features, train_labels, n_samples=docker_size_limit)

# 设置特征工程的标志 【数据预处理是很重要的步骤】
is_features_normal = False
is_labels_encod = False

print("All features and labels uncompressed.")


"""△数据预处理"""
"""
   说明：由于notMNIST图像为灰度图，其像素值的范围为[0,255]
   处理1：需要将图像的像素值范围规划到[0.1,0.9]，对应[a,b]
         使用方法——Min-Max Scaling——X’=a + (X-Xmin)(b-a)/(Xmax-Xmin)
"""
def normalize_grayscale(image_data):
    a = 0.1
    b = 0.9
    grayscale_min = 0
    grayscale_max = 255
    return a + ((image_data - grayscale_min) * (b - a) / (grayscale_max - grayscale_min))

#  ****************************************************************************
# 测试是否正确缩小了像素值的范围
np.testing.assert_array_almost_equal(
    normalize_grayscale(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 255])),
    [0.1, 0.103137254902, 0.106274509804, 0.109411764706, 0.112549019608, 0.11568627451, 0.118823529412, 0.121960784314,
     0.125098039216, 0.128235294118, 0.13137254902, 0.9],
    decimal=3)
np.testing.assert_array_almost_equal(
    normalize_grayscale(np.array([0, 1, 10, 20, 30, 40, 233, 244, 254,255])),
    [0.1, 0.103137254902, 0.13137254902, 0.162745098039, 0.194117647059, 0.225490196078, 0.830980392157, 0.865490196078,
     0.896862745098, 0.9])
#  ****************************************************************************

# 如果没有进行数据处理
if not is_features_normal:
    train_features = normalize_grayscale(train_features)
    test_features = normalize_grayscale(test_features)
    is_features_normal = True

print('Tests Passed!')

"""
   说明：LabelBinarizer()可以将标称型数据进行数值化，数字范围从0开始，并且将label转换为一个列向量；
        若要还原之前的label，需要使用函数inverse_transform()
"""
"""处理标签"""
if not is_labels_encod:
    # 应用one-hot 编码
    encoder = LabelBinarizer()
    encoder.fit(train_labels)
    train_labels = encoder.transform(train_labels)
    test_labels = encoder.transform(test_labels)

    # 转换为float32类型
    train_labels = train_labels.astype(np.float32)
    test_labels = test_labels.astype(np.float32)
    is_labels_encod = True

print('Labels One-Hot Encoded')

# 确保以及进行了数据预处理
assert is_features_normal, 'You skipped the step to normalize the features'
assert is_labels_encod, 'You skipped the step to One-Hot Encode the labels'


"""
   说明：sklearn的train_test_split可以随机划分训练集和测试集；是交叉验证中常用的函数。
         其中，参数test_size：表示测试集占比；
               参数random_state：是随机数种子，实质为该组随机数的编号；
                   在需要重复试验的时候，保证得到一组一样的随机数；
                   每次都填1，其他参数一样的情况下得到的随机数组相同；但填0或不填，每次得到不一样的随机数组。
"""
train_features, valid_featurs, train_labels, valid_labels = train_test_split(
    train_features,
    train_labels,
    test_size=0.05,
    random_state=832289)

print('Training features and labels randomized and split.')

# 保存每个处理好的数据
"""
   说明1：pickle模块实现了基本的数据序列化和反序列化——二进制形式；
         通过该模块的序列化操作可以将程序中运行的对象信息保存到文件(.pkl)中去，永久存储[不能直接打开预览]；
         而通过反序列化操作，能够从文件中创建上一次程序保存的对象。
         1. 序列化操作有：
            pickle.dump()——pickle.dump(obj, file, protocol=None,*,fix_imports=True)
            Pickler(file, protocol).dump(obj)
         2. 反序列化操作有：
            pickle.load()
            Unpickler(file).load()
   拓展：python的另一个序列化标准模块json，保存的文件可以直接打开查看；
   说明2：try...except模块类似于Java中的异常try...catch块
"""
pickle_file = 'notMNIST.pickle'
if not os.path.isfile(pickle_file):
    print('Saving data to pickle file..')
    try:
        with open('notMNIST.pickle', 'wb') as pfile:   # 以二进制形式写入'wb'
            pickle.dump(
                {
                    'train_dataset':train_features,
                    'train_labels':train_labels,
                    'valid_dataset':valid_featurs,
                    'valid_labels':valid_labels,
                    'test_dataset':test_features,
                    'test_labels':test_labels,
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

print('Data cached in pickle file.')

# ----------------------------------------------------------------------------------------------------------------------

# 利用保存的pickle文件重新加载数据
pickle_file = 'notMNIST.pickle'
with open(pickle_file, 'rb') as f:
  pickle_data = pickle.load(f)
  train_features = pickle_data['train_dataset']
  train_labels = pickle_data['train_labels']
  valid_features = pickle_data['valid_dataset']
  valid_labels = pickle_data['valid_labels']
  test_features = pickle_data['test_dataset']
  test_labels = pickle_data['test_labels']
  del pickle_data  # 释放内存

print('Data and modules loaded.')

# ----------------------------------------------------------------------------------------------------------------------

"""使用TensorFlow框架建立单层的神经网络模型(仅含一个输入层和一个输出层)"""
"""
   tensor类型说明：
         1. features(train_features/valid_features/test_features)
            为Placeholder tensor；因为特征的数值可以在训练时传入，不需要初始化；且值固定
         2. labels(train_labels/valid_labels/test_labels)
            为Placeholder tensor；
         3. weights
            为Variable tensor；用正态分布的值来初始化
         4. biases
            为Variable tensor；用0来初始化
   △笔记：
     tf.placeholder和tf.variable的区别——
         1. tf.placeholder在声明的时候不需要初始化的数值，只需要声明类型和维数，例如
                       x = tf.placeholder(tf.float32, shape=(None, 1024))；
            tf.placeholder是为了方便定义神经网络结构，所以可以看作是符号变量；
            tf.placeholder通常是在训练session开始后，存放输入样本的。
            
         2. tf.Variable在声明的时候必须要有初始的数值。例如
                    weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),name="weights")
                    biases = tf.Variable(tf.zeros([200]), name="biases")
            tf.Variable通常是存放weight和bias，然后会不停地被更新，所以说是variable。
"""
features_count = 784
labels_count = 10

# 设置特征和标签的tensor对象
features = tf.placeholder(tf.float32)
labels = tf.placeholder(tf.float32)

# 设置权重和偏置的tensor对象
weights = tf.Variable(tf.truncated_normal((features_count, labels_count)))
biases = tf.Variable(tf.zeros(labels_count))

#  ****************************************************************************
# 测试特征、标签、权重和偏置的tensor对象和形状以及特征和标签的类型是否正确
from tensorflow.python.ops.variables import Variable

assert features._op.name.startswith('Placeholder'), 'features must be a placeholder'
assert labels._op.name.startswith('Placeholder'), 'labels must be a placeholder'
assert isinstance(weights, Variable), 'weights must be a TensorFlow variable'
assert isinstance(biases, Variable), 'biases must be a TensorFlow variable'

assert features._shape == None or (
    features._shape.dims[0].value is None and
    features._shape.dims[1].value in [None, 784]), 'The shape of features is incorrect'
assert labels._shape  == None or (
    labels._shape.dims[0].value is None and
    labels._shape.dims[1].value in [None, 10]), 'The shape of labels is incorrect'
assert weights._variable._shape == (784, 10), 'The shape of weights is incorrect'
assert biases._variable._shape == 10, 'The shape of biases is incorrect'

assert features._dtype == tf.float32, 'features must be type float32'
assert labels._dtype == tf.float32, 'labels must be type float32'
#  ****************************************************************************

# Feed dicts for training, validation, and test session——训练集，验证集和测试集
train_feed_dict = {features: train_features, labels: train_labels}
valid_feed_dict = {features: valid_featurs, labels: valid_labels}
test_feed_dict = {features: test_features, labels: test_labels}


"""
   说明：tf.matmul()实现矩阵乘法，使用时应注意顺序
"""
# 实现线性函数WX+b
logits = tf.matmul(features, weights) + biases

# 得到预测值
prediction = tf.nn.softmax(logits)

"""
    说明：tf.reduce_sum()函数实现矩阵元素的求和，原reduction_indices参数指定求和的维度，
                                               0表示按列求和，1表示按行求和；而[0,1]会处理为一个数；
          现在axis也能表示相同的功能
"""
# 交叉熵函数
cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), reduction_indices=1)

# 得到损失函数——对交叉熵误差取均值
loss = tf.reduce_mean(cross_entropy)

# 创建初始化所有变量(variables)的操作
init = tf.global_variables_initializer()

#  ****************************************************************************
# 测试
with tf.Session() as session:
    session.run(init)    # 初始化总在最前
    session.run(loss, feed_dict=train_feed_dict)  # 得到训练集的损失
    session.run(loss, feed_dict=valid_feed_dict)  # 得到验证集的损失
    session.run(loss, feed_dict=test_feed_dict)   # 得到测试集的损失
    biases_data = session.run(biases)

assert not np.count_nonzero(biases_data), 'biases must be zeros'

print('Tests Passed!')
#  ****************************************************************************

"""
   说明：tf.argmax是对矩阵按行或列返回最大值的索引；0表示按列，1表示按行[貌似和numpy中的操作相反]
"""
# is_correct_prediction变量记录预测正确的个数
is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
# 计算预测精度——取均值
accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

print('Accuracy function created.')

# ----------------------------------------------------------------------------------------------------------------------

"""调整超参数，使神经网络模型的精度达到理想值"""
"""
   说明：要调整的参数主要为epochs和learning rate；分别进行两种设置——
        1. 保持epochs为1，依次更改learning rate的值：0.8, 0.5, 0.1, 0.05, 0.01
        2. 保持learning rate为0.2，依次更改epochs的值：1, 2, 3, 4, 5
   这里使用验证集来进行超参数的调整；
"""
# 设定batch的大小
batch_size = 128

# 不断调整超参数
epochs = 20
learning_rate = 0.1
# learning_rate = 0.5
# learning_rate = 0.1
# learning_rate = 0.05
# learning_rate = 0.01

# learning_rate = 0.2
# epochs = 2
# epochs = 3
# epochs = 4
# epochs = 5

# 梯度下降——用TensorFlow来进行优化
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# 利用验证集来调整超参数；初始化验证集的精确度
validation_accuracy = 0.0

# 对损失和精度绘图
log_batch_step = 50
batches = []
loss_batch = []
train_acc_batch = []
valid_acc_batch = []

"""
   说明：math.ceil()函数对一个数进行向上取整
"""
with tf.Session() as session:
    session.run(init)
    batch_count = int(math.ceil(len(train_features)/batch_size))   # batch的个数——分几批训练

    for epoch_i in range(epochs):
        # 进度条
        batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')

        # 循环训练
        for batch_i in batches_pbar:
            # 依次在训练集中取特定batch_size(这里为128)大小的小批量数据进行训练
            batch_start = batch_i * batch_size   # 该批batch开始的索引
            batch_features = train_features[batch_start:batch_start + batch_size]
            batch_labels = train_labels[batch_start:batch_start + batch_size]

            # 启动优化器并获得损失值
            _, l = session.run(
                [optimizer, loss],
                feed_dict={features: batch_features, labels: batch_labels})

            # 每隔50次选取batch的训练，计算一次训练精度和验证精度
            if not batch_i % log_batch_step:
                training_accuracy = session.run(accuracy, feed_dict=train_feed_dict)
                validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

                # △这里不是很理解
                previous_batch = batches[-1] if batches else 0
                batches.append(log_batch_step + previous_batch)

                # 将该批损失值，训练精度和验证精度添加到列表中
                loss_batch.append(l)
                train_acc_batch.append(training_accuracy)
                valid_acc_batch.append(validation_accuracy)

        # 一次迭代中所有的训练集都按batch分批训练完后，用验证集来计算精度
        validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

    # 绘图
    loss_plot = plt.subplot(211)
    loss_plot.set_title('Loss')
    loss_plot.plot(batches, loss_batch, 'g')
    loss_plot.set_xlim([batches[0], batches[-1]])
    acc_plot = plt.subplot(212)
    acc_plot.set_title('Accuracy')
    acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
    acc_plot.plot(batches, valid_acc_batch, 'x', label='Validation Accuracy')
    acc_plot.set_ylim([0, 1.0])
    acc_plot.set_xlim([batches[0], batches[-1]])
    acc_plot.legend(loc=4)
    plt.tight_layout()
    plt.show()

    print('Validation accuracy at {}'.format(validation_accuracy))

# ----------------------------------------------------------------------------------------------------------------------

"""利用测试集测试网络模型的精度"""
test_accuracy = 0.0

with tf.Session() as session:
    session.run(init)
    batch_count = int(math.ceil(len(train_features) / batch_size))

    for epoch_i in range(epochs):

        # 进度条
        batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i + 1, epochs), unit='batches')

        # 重复训练过程
        for batch_i in batches_pbar:
            # 得到训练集的小批量数据
            batch_start = batch_i * batch_size
            batch_features = train_features[batch_start:batch_start + batch_size]
            batch_labels = train_labels[batch_start:batch_start + batch_size]

            # 运行优化器，优化网络模型
            _ = session.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})

        # 全部训练集都用来优化网络模型后，利用测试集计算精度
        test_accuracy = session.run(accuracy, feed_dict=test_feed_dict)

# 确保测试精度大于80%
assert test_accuracy >= 0.80, 'Test accuracy at {}, should be equal to or greater than 0.80'.format(test_accuracy)

print('Nice Job! Test Accuracy is {}'.format(test_accuracy))


