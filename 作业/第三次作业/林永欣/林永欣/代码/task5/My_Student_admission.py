import pandas as pd
import numpy as np


# 激活函数，使用sigmoid函数，1/(1+exp(-x))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# sigmoid函数的导数为, y*(1-y)
def sigmoid_derivation(x):
    return sigmoid(x) * (1-sigmoid(x))


# 二分类的交叉熵计算公式
def cross_entropy(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)


# 计算反向传播误差的公式
def error_term_formula(y, output):
    return (y-output) * output * (1 - output)


# 将数据分成特征和标签
def separated_xy(data):
    y_data = data['admit']
    x_data = data.drop('admit', axis=1)
    return x_data, y_data


# 将数据分成训练集和测试集
def separated_data(data):
    sample = np.random.choice(data.index, size=int(len(data) * 0.8), replace=False)  # 交叉验证取数据
    return data.iloc[sample], data.drop(sample)


# 训练数据的过程
def train_algorithm(features, targets, num_epochs, learn_rate):
    np.random.seed(50)  # 随机生成50次一样的数，方便调试

    n_records, n_features = features.shape  # 数据集的大小
    last_loss = None

    # 使用标准差为1/sqrt(n)的高斯分布随机生成值效果比较好，此值为Xavier初始值
    weights = np.random.normal(scale=1 / np.sqrt(n_features), size=n_features)

    # 循环num_epochs次训练
    for i in range(num_epochs):
        del_w = np.zeros(weights.shape)
        for x, y in zip(features.values, targets):
            result = sigmoid(np.dot(weights, x))  # 计算经过sigmoid激活函数后的值
            error_term = error_term_formula(y, result)  # 计算单个数据yi的反向传播误差值
            del_w += error_term * x  # 计算所有数据的梯度
        weights += learn_rate * del_w / n_records  # 以一定的学习速率更新权重

        # 打印训练的误差函数变变化过程
        if i % 100 == 0:
            out = sigmoid(np.dot(features, weights))  # 计算预测值
            loss = 0.5 * np.mean((out - targets) ** 2)  # 使用均方误差作为损失函数
            print("第{}个:".format(i))
            if last_loss and last_loss < loss:
                print("损失函数的值: ", loss, "  注意：损失函数在增大")
            else:
                print("损失函数的值: ", loss)
            last_loss = loss
            print("=========")
    print("训练完毕!")
    return weights  # 返回训练后的权重


# 缩放处理数据，控制在0-1范围内
def disposal_data(data):
    data['gre'] = data['gre'] / 800  # 测试分数的范围大概是200-800,缩放到0-1范围内
    data['gpa'] = data['gpa'] / 4.0  # 成绩的范围是1.0-4.0,缩放到0-1范围内
    return data


# 对数据中rank列进行one_hot编码
def one_hot_encoder(data_file):
    # 使用pandas中的get_dummies函数来进行one_hot编码
    one_hot_data = pd.concat([data_file, pd.get_dummies(data_file['rank'], prefix='rank')], axis=1)
    one_hot_data = one_hot_data.drop('rank', axis=1)  # 删除数据集中rank这一列
    return one_hot_data


def main():
    num_epochs = 1000  # 迭代次数
    learn_rate = 0.1  # 学习速率，每次权重变化的幅度

    data_file = pd.read_csv(r'F:\项目架构\2 神经网络\2自己动手实现第一个神经网络\student_data.csv')  # 读取csv文件
    one_hot_data = one_hot_encoder(data_file)  # rank特征不好表示，转换成one_hot编码形式
    data = disposal_data(one_hot_data)  # 对数据进行缩放，把范围控制在0-1区间，降低训练难度
    train_data, test_data = separated_data(data)  # 把数据分成数据集和测试集
    x_train_data, y_train_data = separated_xy(train_data)  # 分离出训练集的特征和标签
    x_test_data, y_test_data = separated_xy(test_data)  # 分离出测试集的特征和标签
    post_train_weights = train_algorithm(x_train_data, y_train_data, num_epochs, learn_rate)  # 训练神经网络
    test_out = sigmoid(np.dot(x_test_data, post_train_weights))  # 用训练好的模型预测测试集数据的结果
    predictions = test_out > 0.5  # 如果大于0.5代表结果是1，反之代表0
    accuracy = np.mean(predictions == y_test_data)  # 匹配预测的结果与实际结果，求平均值为预测精度
    print("模型预测精度: {:.3f}".format(accuracy))  # 打印模型预测精度


if __name__ == '__main__':
    main()
