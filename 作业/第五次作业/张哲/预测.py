import pandas as pd
import numpy as np

#####################################################
# 问题
# 没有引入正则项，成本函数由err=1/2 * ((y-y_hat)^2)修改为
# L=-( ylog(a)+(1-y)log(1-a) ),dz2=dL/da * da/dz
# 不同的误差函数影响待验证
# 预测的准确率跳动，跟初始权重有关
#####################################################
data = pd.read_csv(r'C:\Users\zhangwenqi\Desktop\test\AI_UAV\2 神经网络\2自己动手实现第一个神经网络\student_data.csv',
                   engine='python')
# python3 需要engine=‘python’
#####################################################
## 1 分类数据
x = np.array(data[["gre", "gpa"]])  # 400x2
y = np.array(data["admit"])  # 1x400
onehot = np.array(data["rank"])  # 1x400
data_1 = data[data["rank"] == 1]
# print(data["rank"]==2)
# print(len(data_1))
# print(data_1)
data_2 = data[data["rank"] == 2]
data_3 = data[data["rank"] == 3]
data_4 = data[data["rank"] == 4]
# tt 是 one-hot编码400*3
tt = pd.get_dummies(data['rank'])
# print(tt)
# axis=1 表示 合并‘列’
data = pd.concat([data, tt], axis=1)  # 400*8

####################################################
## 2 缩放
data['gre'] = data['gre'] / (np.max(data['gre']))
data['gpa'] = data['gpa'] / (np.max(data['gpa']))

## one hot编码+分割
data_one_hot = data.drop('rank', axis=1)  # 400*7
data_one_hot_test = data_one_hot[0:350]  # 350*7
data_one_hot_pred = data_one_hot[351:400]  # 50*7

# data.columns 从0开始算，rank是第5列，
# data_one_hot.values是dataframe里取值
###################################################
# 激活函数  采用sigmod函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# 激活函数的导数
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))
# 阈值函数
def f(z):
    if sigmoid(z) >= 0.5:
        return 1
    else:
        return 0
## 3 权重与学习率
m_one_hot, n_one_hot = data_one_hot_test.shape # 训练用
m_p, n_p = data_one_hot_pred.shape # 预测用
############################################################
## 3.2 网络结构 6*3*1
y = (data_one_hot_test.values[:, 0]).reshape(m_one_hot, 1)  # 350*1
x = data_one_hot_test.values[:, 1:7]  # 350*6
print(x)
# 预测集
y_p = (data_one_hot_pred.values[:, 0]).reshape(m_p, 1)  # 350*1
x_p = data_one_hot_pred.values[:, 1:7]  # 350*6
# 6*3*1网络结构
w1_hidden = 6
w2_hidden = 1
# 参数初始
# w1表示输入6-第一层3，w2表示第一层3-输出1
w1 = np.random.randn(n_one_hot - 1, w1_hidden)  # 6*3
w2 = np.random.randn(w1_hidden, 1)  # 3*1
dw1 = np.zeros(((n_one_hot - 1), w1_hidden))
dw2 = np.zeros((w1_hidden, 1))
b = np.zeros((1, 2))
db1 = 0
db2 = 0
rate = 0.001
step = 4000
J = 0
#
for i in range(step):
    # 正向传播
    z1 = np.dot(x, w1) + b[0, 0]  # 350*3
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b[0, 1]  # 1
    a2 = sigmoid(z2)
    # 反向传播
    # 3-1层
    dz2 = a2 - y  # 350*1
    ## L=-( ylog(a)+(1-y)log(1-a) ),dz2=dL/da * da/dz
    dw2 = 1 / w1_hidden * np.dot(a1.T, dz2)  # 3*1
    db2 = 1 / w1_hidden * np.sum(dz2)
    w2 = w2 - rate * dw2
    b[0, 1] = b[0, 1] - rate * db2

    # 6-3层
    dz1 = dz2 * sigmoid_prime(np.dot(x, w1) + b[0, 0])  # 350*3
    dw1 = 1 / m_one_hot * np.dot(x.T, dz1)  # 6*3
    db1 = 1 / m_one_hot * np.sum(dz1)
    w1 = w1 - rate * dw1
    b[0, 0] = b[0, 0] - rate * db1
    # print(w1)
    # print(w2)
    # print(b)

###################################################
# 4 预测
acc = 0
aa1 = sigmoid(np.dot(x_p, w1) + b[0, 0])
aa2 = sigmoid(np.dot(aa1, w2) + b[0, 1])

flag = aa2 > 0.5
acc = np.mean(flag == y_p)

print("准确率=%f" % (acc *100))

###################################################
