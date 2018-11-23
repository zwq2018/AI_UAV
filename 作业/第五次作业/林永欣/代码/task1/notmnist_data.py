import os
import pickle  # 它可以将对象转换为一种可以传输或存储的格式 序列化过程将文本信息转变为二进制数据流

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
from tqdm import tqdm
from zipfile import ZipFile


# 解压图片,获取数据特征和标签
def uncompress_features_labels(file):
    features = []
    labels = []
    with ZipFile(file) as zipf:
        # Progress Bar
        filenames_pbar = tqdm(zipf.namelist(), unit='files')

        # Get features and labels from all files
        for filename in filenames_pbar:
            # Check if the file is a directory
            if not filename.endswith('/'):
                with zipf.open(filename) as image_file:
                    image = Image.open(image_file)
                    image.load()

                    feature = np.array(image, dtype=np.float32).flatten()

                    # Get the the letter from the filename.  This is the letter of the image.
                label = os.path.split(filename)[1][0]  # 标签是图片文件名

                features.append(feature)
                labels.append(label)
    return np.array(features), np.array(labels)


# 从训练集数据中获取特征和标签
train_features, train_labels = uncompress_features_labels(
    r'D:\中科大软院\数据集\notMNIST_train.zip')  # features是(210000, 784)  labels是(210000,)
# print(train_features.shape)
# print(train_labels.shape)

# 从测试集数据中获取特征和标签
test_features, test_labels = uncompress_features_labels(r'D:\中科大软院\数据集\notMNIST_test.zip')

# Limit the amount of data to work with a docker container
docker_size_limit = 150000
train_features, train_labels = resample(train_features, train_labels,
                                        n_samples=docker_size_limit)  # resample函数是重新采样，降低了数据量

# Set flags for feature engineering.  This will prevent you from skipping an important step.
is_features_normal = False
is_labels_encod = False


# 归一化,将灰度值范围设置为0.1-0.9
def normalize_grayscale(image_data):
    a = 0.1
    b = 0.9
    min = 0
    max = 255
    return a+(((image_data-min)*(b-a))/(max-min))


# 归一化,将灰度值范围设置为0.1-0.9
if not is_features_normal:
    train_features = normalize_grayscale(train_features)
    test_features = normalize_grayscale(test_features)
    is_features_normal = True


# 对A-J 10个字母进行编码
if not is_labels_encod:
    # 将标签转换为数字，并应用One-Hot编码
    encoder = LabelBinarizer()  # LabelBinarizer是sklearn库里的数值便签二值化的工具
    encoder.fit(train_labels)
    train_labels = encoder.transform(train_labels)
    test_labels = encoder.transform(test_labels)

    # 转换成float32格式
    train_labels = train_labels.astype(np.float32)
    test_labels = test_labels.astype(np.float32)
    is_labels_encod = True


# 分割数据,测试集和验证集
train_features, valid_features, train_labels, valid_labels = train_test_split(
    train_features,
    train_labels,
    test_size=0.05,  # 样本占比
    random_state=832289)  # 随机种子


# 将数据文件单独保存为pickle文件
pickle_file = 'notMNIST.pickle'
if not os.path.isfile(pickle_file):  # 二进制数据流
    print('Saving data to pickle file...')
    try:
        with open(r'D:\中科大软院\数据集\notMNIST.pickle', 'wb') as pfile:
            pickle.dump(
                {
                    'train_dataset': train_features,
                    'train_labels': train_labels,
                    'valid_dataset': valid_features,
                    'valid_labels': valid_labels,
                    'test_dataset': test_features,
                    'test_labels': test_labels,
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
