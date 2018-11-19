import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def data_process():
    rides = pd.read_csv('day.csv')
    # plt.figure()
    # plt.plot(rides['dteday'],rides['cnt'])
    # rides[:24*10].plot(x='dteday', y='cnt')
    # plt.show()
    dummy_fields = ['season', 'weathersit', 'mnth', 'weekday'] #增加了4+4+12+24+7=51列
    for each in dummy_fields:
        dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
        rides = pd.concat([rides, dummies], axis=1)

    fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                      'weekday', 'atemp', 'mnth', 'workingday'] #17+51-9=59
    data = rides.drop(fields_to_drop, axis=1)
    data.head()
    #数据标准化归一化处理（这里采用了z-score标准化方法）
    quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
    # Store scalings in a dictionary so we can convert back later
    scaled_features = {}
    for each in quant_features:
        mean, std = data[each].mean(), data[each].std()
        scaled_features[each] = [mean, std]
        data.loc[:, each] = (data[each] - mean)/std
    #将样本拆分为训练集、测试集和验证集
    test_data = data[-21:]#将最后21天的数据作为测试样本

    # Now remove the test data from the data set
    data = data[:-21]
    # Separate the data into features and targets
    target_fields = ['cnt', 'casual', 'registered']
    features, targets = data.drop(target_fields, axis=1), data[target_fields]
    #将Dataframe类型数据转换为ndarray数据
    features = features.values
    targets = targets.values
    # targets = targets[:,0]
    #转换数据类型
    features = features.astype('float32')
    targets = targets.astype('float32')

    targets = targets[:,0] #以日借用量为学习目标
    #对输入数据进行主成分分析和降维操作
    pca = PCA()
    pca.fit(features)
    components = pca.components_  # 返回样本空间的各个特征向量
    e_v_r = pca.explained_variance_ratio_  # 返回各个成分各自的方差百分比(也称贡献率）
    # print(e_v_r)
    #选取占比90%的15个特征
    pca = PCA(15)
    pca.fit(features)
    features = pca.transform(features)


    test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]#测试样本

    train_features, train_targets = features[:-60], targets[:-60]
    val_features, val_targets = features[-60:], targets[-60:]
    # print(np.shape(val_features),np.shape(train_features),np.shape(features))
    Samples_and_targets = {
        "test_features":test_features,
        "test_targets":test_targets,
        'train_features':train_features,
        'train_targets':train_targets,
        'val_features':val_features,
        'val_targets':val_targets
    }
    return Samples_and_targets