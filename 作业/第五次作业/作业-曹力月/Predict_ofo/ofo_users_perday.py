import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
   加载数据集，并对数据进行预处理
   INFO：2011/1/1-2012-12-31期间每天每小时的骑车人数，每小时租金，以及温度、湿度和风速等信息;
         骑车用户分为临时用户和注册用户，cnt列为骑车用户数汇总列
"""
rides = pd.read_csv('bike-sharing-dataset/hour.csv')
# print(rides)

# 选取大致为前十天的数据，以dteday数据为x轴，cnt数据为y轴，绘制二维图表
# day_number = rides[:24*10].plot(x='dteday', y='cnt')
# plt.show(day_number)

# 对一些如季节、天气、月份等分类变量创建二进制虚拟变量（哑变量）
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)   # 将新列与原来的数据集合并

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)   # 删除上述的无效列
# print(data.head())  # 读取前五条数据

# 将连续变量标准化——使其均值为0，标准差为1
# 保存换算因子，以便在使用网络进行预测时还可以还原数据
quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
scaled_features = {}  # 字典变量：保留换算因子——均值和方差
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()   # 利用某一列属性的数据求均值和方差
    scaled_features[each] = [mean, std]    # 字典中的值可以是一个数组
    # print(scaled_features)
    data.loc[:, each] = (data[each] - mean) / std


"""
   将数据拆分为训练集、测试集和验证集，并拆分特征和标签
   拆分情况：用历史数据进行训练，尝试预测未来数据(验证数据集)；
             选大约最后21天的数据为测试数据集；
             在剩下的数据中选取大约最后60天的数据为验证集。
"""
test_data = data[-21*24:]   # 测试数据集
data = data[:-21*24]    # 训练集和验证集

# 选取标签变量名
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]   # 分别保存特征和目标

# 测试特征，测试目标
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

# 训练特征，训练目标
train_features, train_targets = features[:-60*24], targets[:-60*24]
val_features, val_targets = features[-60*24:], targets[-60*24:]


"""
   训练神经网络模型：随机梯度下降法（SGD）——
                            对于每次训练，都获取随机样本数据，而不是整个数据集；
                    相较于普通梯度下降，训练次数更多，但每次训练时间更短
   过程：
   设置超参数；目标：使训练集上的错误很小且数据不会过拟合——
   1. 选择迭代次数；即训练网络时从训练数据中抽样的批次数量。目标：选择一个使训练损失
                   很低并且验证损失保持中等水平的数字。
   2. 选择学习速率；即权重的更新幅度；建议从0.1开始；【学习速率越低，权重更新的步长就越小，
                   神经网络收敛的时间就越长】；
                   如果网络在与数据拟合时遇到问题，尝试降低学习速率。
   3. 选择隐藏节点的数量；隐藏节点越多，模型的预测结果就越准确：可以尝试不同的隐藏节点的数量。
                   如果隐藏单元的数量太少，那么模型就没有足够的空间进行学习；如果太多，则学习
                   方向就有太多的选择。目标：找到合适的平衡点
"""
import sys
from two_layer_network import *

# 设置超参数
iterations = 100   # 迭代次数
learning_rate = 0.1   # 学习速率
hidden_size = 2    # 隐藏层节点数量
output_size = 1    # 输出层节点数量
n_features = train_features.shape[1]  # 训练数据的特征个数，即输入层的节点数量

# 得到一个神经网络模型
network = TwoLayerNetwork(n_features, hidden_size, output_size, learning_rate)

# 保存训练集和验证集的损失值
losses = {'train':[], 'validation':[]}

# 每次迭代训练
for iter in range(iterations):
    # 每次从训练集中随机选择一部分(batch)进行训练，batch的大小设定为128
    batch = np.random.choice(train_features.index, size=128)
    x, y = train_features.ix[batch].values, train_targets.ix[batch]['cnt']

    network.train(x, y[:, None])  # 利用batch数据集训练神经网络并更新权重参数

    # 利用全部的训练集计算损失误差
    train_loss = mean_square_error(network.predict(train_features).T, train_targets['cnt'].values)
    # 利用验证集计算损失误差
    val_loss = mean_square_error(network.predict(val_features).T, val_targets['cnt'].values)
    sys.stdout.write("\rProgress: {:2.1f}".format(100 * iter/float(iterations))
                     + "% ... Training loss: " + str(train_loss)[:5]
                     + " ... Validation loss: " + str(val_loss)[:5])
    sys.stdout.flush()

    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)

# 绘图
plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
_ = plt.ylim()
# 显示图像
plt.show()


"""获取预测精度：使用测试数据检查网络对数据建模的效果"""
fig, ax = plt.subplots(figsize=(8, 4))

mean, std = scaled_features['cnt']  # 获取处理数据时的换算因子
predictions = network.predict(test_features).T * std + mean    # 将预测值还原到原来的范围

# 绘图
ax.plot(predictions[0], label='Prediction')
ax.plot((test_targets['cnt']*std + mean).values, label='Target')
ax.set_xlim(right=len(predictions))
ax.legend()
fig.show()
# 设置日期的格式
dates = pd.to_datetime(rides.loc[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)
