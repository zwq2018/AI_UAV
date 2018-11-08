import pandas as pd
import numpy as np

"""对原始数据进行one-hot编码"""
def one_hot_encoding(data):
    one_hot_data = pd.concat([data, pd.get_dummies(data['rank'], prefix='rank')], axis=1)
    one_hot_data = one_hot_data.drop('rank', axis=1)
    return one_hot_data

"""数据对齐"""
def data_alignment(one_hot_data):
    processed_data = one_hot_data[:]
    """对数据进行放缩，使数据对齐到[0,1]"""
    processed_data['gre'] = processed_data['gre'] / 800
    processed_data['gpa'] = processed_data['gpa'] / 4.0
    return processed_data

"""数据分割：将原始数据分为训练集和测试集"""
def data_seperation(processed_data):
    """90%为训练数据"""
    sample = np.random.choice(processed_data.index, size=int(len(processed_data) * 0.9), replace=False)
    train_data, test_data = processed_data.iloc[sample], processed_data.drop(sample)
    """将数据分成特征和目标"""
    features = train_data.drop('admit', axis=1)
    targets = train_data['admit']
    features_test = test_data.drop('admit', axis=1)
    targets_test = test_data['admit']
    return features, targets, features_test, targets_test

"""激活函数"""
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

"""梯度计算"""
def error_formula(y, model_output):
    """sigmoid(x)的一阶导数为sigmoid(x) * (1-sigmoid(x))"""
    return (model_output - y) * model_output * (1 - model_output)

"""权值更新"""
def weight_update(x, weights, learn_rate):
    model_output = sigmoid(np.dot(x, weights))  # 当前模型输出
    error = error_formula(y, model_output)      # 当前模型误差
    weights -= learn_rate * error * x           # 梯度下降
    return model_output, weights

"""损失函数"""
def loss_formula(targets, out):
    return (out - targets) ** 2

"""误差跟踪：用于观测模型的误差变化，判断参数更新是否正确"""
def loss_trace(features, targets, weights):
    out = sigmoid(np.dot(features, weights))
    loss = np.mean(loss_formula(targets, out))
    return loss

"""输出当前的模型误差"""
def loss_output(loss, last_loss , i):
    print("Epoch:", i)
    if last_loss and last_loss < loss:
        print("Train loss: ", loss, "  WARNING - Loss Increasing")
    else:
        print("Train loss: ", loss)
    print("=========")

sourcefile = r'F:\项目架构\2 神经网络\2自己动手实现第一个神经网络\student_data.csv'
data = pd.read_csv(sourcefile)
"""数据预处理"""
one_hot_data = one_hot_encoding(data)
processed_data = data_alignment(one_hot_data)
features, targets, features_test, targets_test = data_seperation(processed_data)

"""训练模型"""
epochs = 2000       # 迭代次数
learn_rate = 0.5    # 学习率
n_records, n_features = features.shape
weights = np.random.normal(scale = 1 / n_features ** .5, size = n_features)  #随机生成初始的权值
last_loss = None    # 参数更新前的模型误差，用于模型的误差跟踪
for i in range(epochs):
    for x, y in zip(features.values, targets):
        """模型参数修正"""
        model_output, weights = weight_update(x, weights, learn_rate)
        """模型误差跟踪"""
        if i % (epochs / 10) == 0:
            loss = loss_trace(features, targets, weights)
            loss_output(loss, last_loss, i)
            last_loss = loss
print("Finished training!\n")
print("The trained model :")
print(weights)

"""测试模型"""
test_out = sigmoid(np.dot(features_test, weights))  # 当前模型对测试数据标签的预测
predictions = test_out > 0.5                        # 预测学生是否录取
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))