"""
   该部分对学生数据进行加载和预处理——
         1. 仅用gpa和gre绘制的散点图显示数据并不能因此有很好地分离；加入评级rank后
         观察到评级越低，录取率越高；所以加入rank作为输入的一个特征：一个输入总共获取到3个特征。
         2. 四个评级的数据分别是1,2,3,4，不方便神经网络对其处理，需要对其进行一次one-hot编码
         3. 观察到平时成绩(grades)和考试成绩(test scores)的范围相差很大，需要将这两个特征的数据放在0-1的范围内
         4. 将数据分为训练集和测试集(占总数据的10%)，测试集用以测试神经网络的泛化能力，防止过拟合。
         5. 将数据分为特征和标签(实际上是将输入和输出分离)，以便进行误差的计算和评估
"""
import numpy as np
import pandas as pd


"""用pandas读取学生数据"""
data = pd.read_csv('student_data.csv')

"""
# 仅根据gre和gpa变量绘制数据的散点图
def plot_points(data):
    X = np.array(data[["gre", "gpa"]])
    y = np.array(data["admit"])
    admitted = X[np.argwhere(y == 1)]  # 保存被录取的学生的测试成绩和平时成绩
    rejected = X[np.argwhere(y == 0)]  # 保存被拒绝的学生的测试成绩和平时成绩
    # 绘制散点图
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s=25, color='red', edgecolor='k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s=25, color='cyan', edgecolor='k')
    plt.xlabel('Test (GRE)')
    plt.ylabel('Grades (GPA)')

# 显示散点图
# plot_points(data)
# plt.show()


# 加入评级rank标准来绘制数据图
def plot_points_rank(data):
    data_rank1 = data[data["rank"] == 1]
    data_rank2 = data[data["rank"] == 2]
    data_rank3 = data[data["rank"] == 3]
    data_rank4 = data[data["rank"] == 4]

    # 图片显示
    plot_points(data_rank1)
    plt.title("Rank 1")
    plt.show()
    plot_points(data_rank2)
    plt.title("Rank 2")
    plt.show()
    plot_points(data_rank3)
    plt.title("Rank 3")
    plt.show()
    plot_points(data_rank4)
    plt.title("Rank 4")
    plt.show()


plot_points_rank(data)
"""


"""使用pandas中的get_dummies方法对评级数据进行one-hot编码"""
one_hot_data = pd.concat([data, pd.get_dummies(data['rank'], prefix='rank')], axis=1)
one_hot_data = one_hot_data.drop('rank', axis=1)
# print(one_hot_data[:10])


"""缩放数据——将平时成绩(grades)除以4.0，将测试成绩(test scores)除以800"""
processed_data = one_hot_data[:]
processed_data['gre'] = processed_data['gre']/800
processed_data['gpa'] = processed_data['gpa']/4.0
# print(processed_data[:10])


"""将数据分成训练集和测试集"""
sample = np.random.choice(processed_data.index, size=int(len(processed_data)*0.9), replace=False)
train_data, test_data = processed_data.iloc[sample], processed_data.drop(sample)

# print("Number of training samples is", len(train_data))
# print("Number of testing samples is", len(test_data))
# print(train_data[:10])
# print(test_data[:10])


"""将数据分成特征和标签(输入和输出的分离)"""
features = train_data.drop('admit', axis=1)
targets = train_data['admit']
features_test = test_data.drop('admit', axis=1)
targets_test = test_data['admit']

# print(features[:10])
# print(targets[:10])

