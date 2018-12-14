import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
import pickle
from PIL import Image


#读入图片数据，完成特征化、标签化、归一化

# Load image data as 1 dimensional array
# We're using float32 to save on memory space
def load_pic(file_path):
    labels = []
    features = []
    path_Dir = os.listdir(file_path)
    i = 0
    for allDir in path_Dir:
        filepath = file_path + '/'
        #子文件夹路径
        child = os.path.join("%s%s" % (filepath,allDir))
        for pic in os.listdir(child):
            pic_file_name = child+'/'+str(pic)
            try:
                image = Image.open(pic_file_name)
                image.load()
            except Exception:#读取异常时，直接忽略该样本
                pass
            #将图片特征化为一维向量
            feature = np.array(image, dtype=np.float32).flatten()
            #保持特征尺寸28*28
            feature = np.array(image, dtype=np.float32)
            label = allDir #
            labels.append(label)
            features.append(feature)
            if i%10000==0:
                print('{} pictures has been processed'.format(i))
            i+=1
    return np.array(labels),np.array(features)


#Limit the amount of data to work with a docker container
docker_size_limit = 150000
train_labels,train_features = load_pic(r'F:\uav+ai\data_for_neural_network\MNIST_data\notMNIST_large')
test_labels,test_features = load_pic('H:/AI/查良瑜 作业/task02_tensorflow/notMNIST_small.tar.gz_files/notMNIST_small')
train_labels, train_features = resample(train_labels, train_features, n_samples=docker_size_limit)#resample函数是重新采样，降低了数据量
# Set flags for feature engineering.  This will prevent you from skipping an important step.
is_features_normal = False
is_labels_encod = False

# def normalize_grayscale(image_data): #归一化 将灰度值范围设置为0.1-0.9
#
#     a = 0.1
#     b = 0.9
#     grayscale_min = 0
#     grayscale_max = 255
#     return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )


if not is_features_normal:
    train_features = (train_features/255.0*0.99)+0.01
    test_features = (test_features/255.0*0.99)+0.01
    is_features_normal = True
print('Tests Passed!')

if not is_labels_encod: #对ABCD10个字母编码
    # Turn labels into numbers and apply One-Hot Encoding
    encoder = LabelBinarizer() #LabelBinarizer是sklearn库里的数值便签二值化的工具
    encoder.fit(train_labels)
    train_labels = encoder.transform(train_labels)
    test_labels = encoder.transform(test_labels)

    # Change to float32, so it can be multiplied against the features in TensorFlow, which are float32
    train_labels = train_labels.astype(np.float32)
    test_labels = test_labels.astype(np.float32)
    is_labels_encod = True
#
print('Labels One-Hot Encoded')
#产生训练集和验证集
train_features, valid_features, train_labels, valid_labels = train_test_split(
    train_features,
    train_labels,
    test_size=0.05,
    random_state=832289)

print('Training features and labels randomized and split.')


pickle_file = 'notMNIST.pickle'
if not os.path.isfile(pickle_file):    #判断是否存在此文件，若无则存储
    print('Saving data to pickle file...')
    try:
        with open('notMNIST.pickle', 'wb') as pfile:
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

print('Data cached in pickle file.')

