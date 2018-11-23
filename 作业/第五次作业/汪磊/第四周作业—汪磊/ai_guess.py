import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

# 数据预处理
def preprocessing(data):
    # 从原始数据中提取特征和标签
    feature = data.loc[:, ["feature1", "feature2", "feature3", "feature4"]]
    label = data.loc[:, ["class"]]

    # 数据归一化:feature2和feature4归一化后，feature1和feature3不到这两者的10倍，暂时不做处理
    feature['feature2'] = (feature['feature2'] - feature['feature2'].min()) / (
                feature['feature2'].max() - feature['feature2'].min())
    feature['feature4'] = (feature['feature4'] - feature['feature4'].min()) / (
                feature['feature4'].max() - feature['feature4'].min())

    # 对数据标签进行one-hot编码
    dummy_label = pd.get_dummies(label, columns=['class'])

    return feature, dummy_label

# 激活函数：softmax函数
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

# 损失函数：均方差函数
def mean_square_error(output, label):
    return np.sum(0.5 * (output -label) ** 2, axis=0)

# 训练模型
def model_train(feature, label, learning_rate = 0.1, iteration = 1000):
    n_records, n_feature = feature.shape #n_quantity = 105, n_feature = 4
    weight = np.zeros([feature.shape[1], label.shape[1]]) # 输入层有4个单元，输出层有3个单元，故权重矩阵为4*3
    #last_mean_loss = None    # 记录上一次迭代的误差，用于观察梯度下降过程中误差的变化
    loss = []   # 记录每次迭代的平均误差

    # 样本较少，采用标准梯度下降
    for i in range(iteration):
        current_loss = [] # 记录当前迭代每个数据点的误差
        for x, y in zip(feature.values, label.values):
            output = softmax(np.dot(x, weight)) # 1*4 dot 4*3 = 1*3
            softmax_primer = output * (1 - output)  # 对激活函数求导，1*3
            gradient = np.array((output - y) * softmax_primer)  # 梯度
            weight -= np.dot(x[:, None], gradient[:, None].T) * learning_rate   # 权重更新，注意一维数组向行向量和列向量的转化

            # 记录误差
            error = mean_square_error(output, y)
            current_loss.append(error)
        mean_loss = np.mean(np.array(current_loss)) # 当前迭代的平均误差
        loss.append(mean_loss)

        # 动态地更新学习率
        if i % 10 == 0:
            learning_rate *= 0.9

    # 输出误差变化过程
    plt.plot(loss, label='Training loss')
    plt.legend()
    plt.show()

    return weight

# 测试模型
def model_test(test_feature, test_label, weight):
    prediction  = np.array(softmax(np.dot(test_feature,weight)))    # 45*4 dot 4*3 = 45*3
    # 模型输出的是数据属于三个类别的概率，取其中的最大值将其设为1，其余设为0
    for i in prediction.max(axis=1):    # 采用np.where函数来检索最值的下表
        prediction[np.where(prediction == i)] = 1
    prediction[np.where(prediction != 1)] = 0

    # 精确度计算
    accuracy = np.array(test_label * prediction, dtype=int)
    accuracy = accuracy.sum() / int(len(prediction))
    return accuracy

sourcefile = r'E:\git\AI_UAV\2 神经网络\3bp神经网络\1ai_guess\data.csv'
data = pd.read_csv(sourcefile)
initial_learning_rate = 0.2
iteration = 300 # 由模型训练过程中的误差变化得
feature, lable = preprocessing(data)
# 原始数据集较小，将训练集和测试集的比例设为7:3
train_feature, test_feature, train_label, test_label = train_test_split(feature, lable, test_size=0.3)
weight = model_train(train_feature, train_label, initial_learning_rate, iteration)
accuracy = model_test(test_feature, test_label, weight)
print("Model accuracy: {:.3f}".format(accuracy))

# 模型使用
valid_file = r"E:\git\AI_UAV\2 神经网络\3bp神经网络\1ai_guess\images\validations.csv"
valid_data = pd.read_csv(valid_file)
# 数据预处理
valid_data['feature2']=(valid_data['feature2'] - valid_data['feature2'].min()) / (
        valid_data['feature2'].max() - valid_data['feature2'].min())
valid_data['feature4']=(valid_data['feature4'] - valid_data['feature4'].min()) / (
        valid_data['feature4'].max() - valid_data['feature4'].min())
# 验证
valid_output = softmax(np.dot(valid_data, weight))
for i in valid_output.max(axis=1):
    valid_output[np.where(valid_output == i)] = 1;
valid_output[np.where(valid_output != 1)] = 0;
print("Classify result:")
print(valid_output)