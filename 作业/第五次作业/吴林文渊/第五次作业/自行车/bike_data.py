import numpy as np
import pandas as pd



# one-hot编码
def one_hot(rides):
    dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
    for each in dummy_fields:
        dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
        rides = pd.concat([rides, dummies], axis=1)

    fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 'weekday', 'atemp', 'mnth', 'workingday', 'hr']
    data = rides.drop(fields_to_drop, axis=1)
    return data


# 连续变量标准化，即转换和调整变量，使它们的均值为 0，标准差为 1。
def standard_data(data):
    scaled_features = {}
    quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
    for each in quant_features:
        mean, std = data[each].mean(), data[each].std()  # 平均值 方差
        scaled_features[each] = [mean, std]
        data.loc[:, each] = (data[each] - mean) / std  # 每一列
    return scaled_features,data


# 分离数据
def data_test_train_val(data):
    # 目标值
    target_fields = ['cnt', 'casual', 'registered']

    # 最后21天的数据作为测试数据
    test_data = data[-21 * 24:]
    test_features = test_data.drop(target_fields, axis=1)
    test_targets = test_data[target_fields]

    # 继续拆分为训练集和测试集
    data = data[:-21 * 24]
    features = data.drop(target_fields, axis=1)
    targets = data[target_fields]

    # 训练集
    train_features = features[:-60 * 24]
    train_targets = targets[:-60 * 24]

    # 验证集
    val_features = features[-60 * 24:]
    val_targets = targets[-60 * 24:]

    return test_features, test_targets, train_features, train_targets, val_features, val_targets