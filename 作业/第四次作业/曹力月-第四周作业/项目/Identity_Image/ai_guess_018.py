import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # seaborn作为matplotlib的补充，导入后会覆盖matplotlib的默认作图风格
from sklearn.cross_validation import train_test_split   # 用sklearn包中的方法对训练集和测试集进行划分
from functions import *

"""数据预处理"""
# 加载数据
dataset = pd.read_csv('data.csv')  # dataset为DataFrame格式
# print(dataset)

# 将数据集随机打乱
dataset = dataset.sample(frac=1).reset_index(drop=True)   # 打乱后索引仍按照正常的排序
# print(dataset)

# 可视化数据分析
# 1.使用matplotlib绘制二维变量图——仅选取两个特征
"""
def plot_points(dataset):
    X = np.array(dataset[["feature1", "feature2"]])  # X保存特征，先绘制二维图（选取feature1和feature2）
    y = np.array(dataset["class"])  # y保存类别

    people = X[np.argwhere(y == 0)]  # 取出图片为人时的特征值数组
    cat = X[np.argwhere(y == 1)]  # 取出图片为猫时的特征值数组
    dog = X[np.argwhere(y == 2)]  # 取出图片为狗时的特征值数组

    # 对于类别为人，猫，狗，分别绘制不同颜色的数据点——对应于同一张二维坐标图
    plt.scatter([s[0][0] for s in people], [s[0][1] for s in people], s=25, color='red', edgecolor='k')
    plt.scatter([s[0][0] for s in cat], [s[0][1] for s in cat], s=25, color='cyan', edgecolor='k')
    plt.scatter([s[0][0] for s in dog], [s[0][1] for s in dog], s=25, color='yellow', edgecolor='k')

    plt.xlabel('Feature_1')
    plt.ylabel('Feature_2')


plot_points(dataset)
plt.show()  # 绘制散点图
"""

# 2. 使用seaborn绘制多变量图——选取全部四个特征
sns.pairplot(dataset, hue='class', vars=["feature1", "feature2", "feature3", "feature4"])  # hue : 使用指定变量为分类变量画图；vars : 与data使用，否则使用data的全部变量
# plt.show()  # 仍然使用matplotlib的显示函数

# 将训练集拆分为输入和标签
data = dataset.iloc[0: 150, 0: 4]  # 输入
# print(data)
label = dataset.iloc[0: 150, 4]  # 标签
# print(label)

# 数据标准化——将特征的值规划到相同的范围内，避免数据的偏差，使神经网络易于处理
data['feature2'] = (data['feature2'] - data['feature2'].min()) / (data['feature2'].max() - data['feature2'].min())
data['feature4'] = (data['feature4'] - data['feature4'].min()) / (data['feature4'].max() - data['feature4'].min())

# 将类别[标签]进行One-hot编码，以实现softmax的概率分布
one_hot_label = pd.get_dummies(label)
print(one_hot_label)


"""将处理好的数据分为训练集和测试集,其中测试集的大小占总数据的30%"""
# 使用sklearn.cross_validation.train_test_split方法，暂取随机数种子为1，以在重复试验时，得到相同的随机数组
train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.3, random_state=1)


"""训练神经网络"""
# 设定超参数
iters_num = 10000   # 设定更新(迭代)次数
train_size = train_data.shape[0]    # 训练数据的个数
print(train_size)
learning_rate = 0.1   # 学习率

# 定义记录学习过程的各个函数列表
train_loss_list = []   # 保存每次更新后损失函数的值
train_acc_list = []    # 保存每次更新后对训练数据的识别精度的值
test_acc_list = []     # 保存每次更新后对测试数据的识别精度的值

#epoch(新纪元)的大小，设定更新多少次之后记录一次识别精度
iter_per_epoch = np.sqrt(train_size)

# 进行训练的函数
def train_neural_network(features, labels):
    # 用标准差为0.01的高斯分布初始化各连接线权重（已知输出的类别数为3——人，猫，狗）
    weights = 0.01 * np.random.randn(train_size, 3)# weights是105*3

    for i in range(iters_num):
        w_gradient = np.zeros(weights.shape)    # w_gradient数组保存所有数据的权重梯度
        for data, label in zip(features.values, labels.values):  # 依次对每个输入数据进行预测和更新
             output = softmax(np.dot(data, weights))   # 得到一个数据的预测值

             error = cross_entropy_error(output, label)  # 使用交叉熵计算误差函数
             train_loss_list.append(error)   # 记录每次的损失函数

             # 得到误差项
             error_term = error_term_formula(data, label, output)
             w_gradient += error_term   # 将单个数据传播得到的误差项添加到数组中

        # 所有训练数据都输入完成
        # 根据得到的梯度数组对权重进行更新
        weights -= learning_rate * w_gradient / train_size   # 取均值

        if i % iter_per_epoch == 0:   # 到达了一个新纪元，输出各项观测值
            for data, label in zip(features.values, labels.values):
                # 测试数据的精度
                train_acc = accuracy(weights, data, label)
                train_acc_list.append(train_acc)

                # 训练数据的精度
                test_acc = accuracy(weights, data, label)
                test_acc_list.append(test_acc)

                # 输出更新次数
                print("第", i, "次迭代")

                # 输出训练精度
                print("train acc, test acc |" + str(train_acc) + ',' + str(test_acc))

                # 对已得到的损失函数值求均值，并进行输出
                loss_mean = np.mean(np.array(train_loss_list))
                print("train loss_mean |", loss_mean)

                print("--------------")

    print("Finished training!")

    # 输出训练后模型的预测结果
    output = softmax(np.dot(data, weights))  # 取得输出值
    output = np.argmax(output, axis=1)  # 得到预测值——概率最大者的索引

    if output == 0:
        print("这张图片是人")
    if output == 1:
        print("这张图片是猫")
    if output == 2:
        print("这张图片是狗")


    """
    # 绘制图像
    plt.title("Solution boundary")
    """
    return weights


# 用images文件夹中的两张图片进行测试，图片的特征已经提取到validations.csv中
def test():
    # 加载数据——仅含特征
    my_features = pd.read_csv('./images/validations.csv')

    # 数据标准化——将特征的值规划到相同的范围内，避免数据的偏差，使神经网络易于处理
    my_features['feature2'] = (my_features['feature2'] - my_features['feature2'].min()) / (my_features['feature2'].max() - my_features['feature2'].min())
    my_features['feature4'] = (my_features['feature4'] - my_features['feature4'].min()) / (my_features['feature4'].max() - my_features['feature4'].min())

    weights = train_neural_network(train_data, train_label)  # 用训练好的神经网络模型来预测

    output = softmax(np.dot(my_features, weights))
    output = np.argmax(output, axis=1)

    if output == 0:
        print("这张图片是人")
    if output == 1:
        print("这张图片是猫")
    if output == 2:
        print("这张图片是狗")


test()


