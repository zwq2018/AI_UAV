import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 数据预处理
def preprocessing(ride_data):
    # one-hot编码：部分特征是离散的，且有多个结果
    dummy_features = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
    for x in dummy_features:
        dummy_column = pd.get_dummies(ride_data[x], prefix=x)
        ride_data = pd.concat([ride_data, dummy_column], axis=1)  # 把因为one-hot编码多出的数据拼接到原始数据中

    # 去掉无关特征和已经被one-hot编码的特征
    # 'instant', 'dteday'为无关特征，'atemp', 'workingday'与其他特征的作用重合
    # 'season', 'weathersit', 'mnth', 'hr', 'weeday'只取one-hot编码后的数值，丢弃原始数值
    drop_feature = ['instant', 'dteday', 'atemp','season', 'weathersit',
                    'weekday', 'mnth', 'workingday', 'hr'] # workingday ?
    data = ride_data.drop(drop_feature, axis=1)

    # 数据标准化：部分特征的数值范围过大, x = (x - μ)/s
    scaled_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
    scaled_features_data = {}   # 保存被one-hot编码前的结果
    for x in scaled_features:
        mean, std = ride_data[x].mean(), ride_data[x].std()
        scaled_features_data[x] = [mean, std]
        data.loc[:, x] = (data[x] - mean) / std

    # 将数据分为训练集、测试集
    # 原始数据按日期排列，有先后的顺序关系，不能随机分割
    train_data = data[: -(60 + 21) * 24]   # 训练集
    valid_data = data[-(60 + 21)*24 : -21*24]     # 验证集
    test_data = data[-21*24 :]             # 测试集

    # 将数据分为输入特征和输出目标
    target_features = ['casual', 'registered', 'cnt']
    train_feature, train_target = train_data.drop(target_features, axis=1), train_data['cnt']
    valid_feature, valid_target = valid_data.drop(target_features, axis=1), valid_data['cnt']
    test_feature, test_target = test_data.drop(target_features, axis=1), test_data['cnt']
    return train_feature, train_target, valid_feature, valid_target, test_feature, test_target, scaled_features_data

# 隐藏层激活函数
def hidden_activation(x):
    return 1 / (1 + np.exp(-x))

# 输出层激活函数
def output_activation(x):
    return x

# 损失函数
def mean_square_error(output, y):
    return 0.5 * (output - y) ** 2

# 前向传播
def forward_pass(feature, weight_input_hidden, weight_hidden_output):
    hidden_output = hidden_activation(np.dot(feature[:, None].T, weight_input_hidden))    # 1*56 dot 56*10
    model_output = output_activation(np.dot(hidden_output, weight_hidden_output)) # 1*10 dot 10*1
    return hidden_output, model_output

# 反向传播
def backpropagation(hidden_output, model_output, feature, target, weight_hidden_output):
    # 输出层到隐藏层
    output_error_input = model_output - target # 反向传播中，输出层的误差输入 = 模型的误差，1*1
    output_error_output = output_error_input * 1 # 输出层的误差输出，输出层激活函数的导数为1，1*1
    gradient_hidden_output = np.dot(hidden_output.T, output_error_output) # 隐藏层到输出层的权重更新，10*1 dot 1*1

    # 隐藏层到输入层
    hidden_error_input = np.dot(output_error_output, weight_hidden_output.T) # 隐藏层的误差输入 1*1 dot 1*10
    hidden_error_output = hidden_error_input * hidden_output * (1 - hidden_output) # 隐藏层的误差输出，隐藏层激活函数的导数为y*(1-y)，1*10
    gradient_input_hidden = np.dot(feature[:, None], hidden_error_output) # 输入层到输出层的权重更新，56*1 dot 1*10

    return gradient_input_hidden, gradient_hidden_output

# 模型训练
def train_model(batch_feature, batch_target, weight_input_hidden, weight_hidden_output, learning_rate):
    loss = []   # 一次迭代中，样本点的误差

    # 权重更新过程
    for x, y in zip(batch_feature, batch_target):
        # 数据前向传播
        hidden_output, model_output = forward_pass(x, weight_input_hidden, weight_hidden_output)
        # 误差反向传播
        gradient_input_hidden, gradient_hidden_output = backpropagation(hidden_output, model_output, x, y, weight_hidden_output)
        # 权重更新
        weight_input_hidden -= learning_rate * gradient_input_hidden
        weight_hidden_output -= learning_rate * gradient_hidden_output

        # 误差计算
        error = mean_square_error(model_output, y)
        loss.append(error)
    mean_loss = np.mean(np.array(loss))     # 一次迭代中，训练集随机样本的平均误差，1*1

    return weight_input_hidden, weight_hidden_output, mean_loss

# 模型验证
def valid_model(valid_feature, valid_target, weight_input_hidden, weight_hidden_output):
    loss1=0
    hidden_output = hidden_activation(np.dot(valid_feature, weight_input_hidden)) # n_val*56 dot 56*10
    model_output = output_activation(np.dot(hidden_output, weight_hidden_output)) # n_val*10 dot 10*1
    y1 = valid_target.values[:, np.newaxis]
    error = mean_square_error(model_output, y1)

    loss1 = np.mean(error)   # 一次迭代中，验证集的平均误差，1*1

    return loss1

# 模型测试
def test_model(test_feature, weight_input_hidden, weight_hidden_output):
    hidden_output = hidden_activation(np.dot(test_feature, weight_input_hidden)) # n_test*56 dot 56*10
    model_output = output_activation(np.dot(hidden_output, weight_hidden_output)) # n_test*10 dot 10*1

    return model_output

sourcefile = r'F:\项目架构\作业\第五次作业\作业-曹力月\Predict_ofo\bike-sharing-dataset\hour.csv'
ride_data = pd.read_csv(sourcefile)
# 数据预处理
train_feature, train_target, valid_feature, valid_target, test_feature, test_target, scaled_features_data = preprocessing(ride_data)
input_node = 56
hidden_node = 10
output_node = 1
learning_rate = 0.01
iteration = 800
# 初始化权重矩阵
weight_input_hidden = np.random.normal(0.0, input_node**-0.5, (input_node, hidden_node))    # 输入层到隐藏层的权重矩阵，56*10
weight_hidden_output = np.random.normal(0.0, hidden_node**-0.5, (hidden_node, output_node)) # 隐藏层到输出层的权重矩阵，10*1
train_loss = []     # 训练集误差
valid_loss = []     # 验证集误差
# 选择模型
for i in range(iteration):
    # 样本较大，采样随机梯度下降：每次迭代随机选择1/1000的数据
    batch_index = np.random.choice(train_feature.index, size=int(train_feature.size / 1000))  # 返回行索引
    batch_feature, batch_target = train_feature.iloc[batch_index].values, train_target.iloc[batch_index].values
    # 模型训练
    weight_input_hidden, weight_hidden_output, current_train_loss= train_model(batch_feature, batch_target, weight_input_hidden, weight_hidden_output, learning_rate)
    # 模型验证
    current_valid_loss = valid_model(valid_feature, valid_target, weight_input_hidden, weight_hidden_output)
    # 误差记录
    train_loss.append(current_train_loss)
    valid_loss.append(current_valid_loss)
    # 更新学习率
    # if i % 20 == 0:
    #     learning_rate *= 0.8
# 输出误差变化过程
plt.plot(train_loss, label="Training 'cnt' loss")
plt.plot(valid_loss, label="Validation 'cnt' loss")
plt.legend()
plt.show()
# 模型测试
mean, std = scaled_features_data['cnt']
test_output = test_model(test_feature, weight_input_hidden, weight_hidden_output)
plt.plot(test_output * std + mean, label="Prediction 'cnt'")
plt.plot(test_target.values * std + mean, label="Initial 'cnt'")    # Dataframe.values返回array
plt.legend()
plt.show()
