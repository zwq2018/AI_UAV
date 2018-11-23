import numpy as np
import pandas as pd

"""
   step1
"""
"""管护数据集"""
def pretty_print_review_and_label(i):
    print(labels[i] + "\t:\t" + reviews[i][:80] + "...")


g = open('sentiment-network\\reviews.txt', 'r')   # 输入
reviews = list(map(lambda x: x[:-1], g.readlines()))  # 将reviews保存为列表：一行为一段评论
g.close()

g = open('sentiment-network\labels.txt', 'r')    # 标签
labels = list((map(lambda x: x[:-1].upper(), g.readlines())))  # 将评价的字母全部变为大写，并保存为列表
g.close()

# print(len(reviews))
# print(reviews[0])
# print(labels[0])

"""
# 建立预测理论——找出数据集间的相关性——如何从输入得到对应的标签
# 认为评论中存在含有明显预测性和相关性的词汇
print("labels.txt \t : \t reviews.txt\n")
pretty_print_review_and_label(2137)
pretty_print_review_and_label(12816)
pretty_print_review_and_label(6267)
pretty_print_review_and_label(21934)
pretty_print_review_and_label(5297)
pretty_print_review_and_label(4998)
"""

"""快速验证预测理论——对评论中出现的词汇进行计数——使用到计数器类Counter"""
from collections import Counter

# 创建计数器对象，类似字典变量
positive_counts = Counter()   # 正面评价的单词计数
negative_counts = Counter()   # 负面评价的单词计数
total_counts = Counter()      # 所有单词计数

# method1：找到正面评价或负面评价中最常见的单词——
#           遍历所有评价，每遇到一个出现在正面评价里的单词，就为该单词的正面评价计数器和总计数器增量；负面同理；
for i in range(len(reviews)):
    if labels[i] == 'POSITIVE':
        for word in reviews[i].split(" "):
            positive_counts[word] += 1
            total_counts[word] += 1
    else:
        for word in reviews[i].split(" "):
            negative_counts[word] += 1
            total_counts[word] += 1

# 分别列出出现在正面评价和负面评价中的单词，并按出现频率由高到低排序
# print(positive_counts.most_common())
# print(negative_counts.most_common())

# 由method1的结果可见，两者之一的常见单词并非有意义的单词
# method2：寻找正面评价里的出现频率比在负面评价里高的单词，以及在负面评价里的出现频率比在正面评价里高的单词——
#           计算单词在正面评价与负面评价中出现次数的【比值】
pos_neg_ratios = Counter()   # 比值计数器
for term, cnt in list(total_counts.most_common()):
    if cnt > 100:    # 通过调整标定的出现次数的大小，可以过滤得到有用的单词数据
        pos_neg_ratio = positive_counts[term] / float(negative_counts[term]+1e-7)
        pos_neg_ratios[term] = pos_neg_ratio

# 查看若干单词的比值的计算结果
# print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
# print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
# print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))

# method3：需要找到一个中心点，从而可以根据数值(比值)来判断单词蕴含多少感情色彩(褒义/贬义)
#           ——以中性值为所有数值的中心，以单词的(正面评价/负面评价的比值)与中性值之差的
#               【绝对值】来衡量单词蕴含的感情色彩的程度；
#           ——在比较绝对值时，以0为中心要比以1为中心容易；
#           ——需要对所有比值进行对数变换，即褒义色彩和贬义色彩程度相同的单词(正面评价/负面评价比)
#               大小相似，但符号相反。
for word, ratio in pos_neg_ratios.most_common():
    pos_neg_ratios[word] = np.log(ratio)

# 查看三种单词的新比值结果
print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))

# 观察更多比值
# print(pos_neg_ratios.most_common())   # 列出所有单词，并按单词与正面评价的相关性排序
# print(list(reversed(pos_neg_ratios.most_common()))[0:30])   # 列出30个与负面评价关系最紧密的单词

# ----------------------------------------------------------------------------------------------------------------------

"""
   step2
"""

"""将文本转换成数字"""
"""创建输入/输出数据"""
# 创建集合vocab，保存出现在评论中的所有单词
vocab = set(total_counts.keys())

# 查看词汇表的大小
vocab_size = len(vocab)
# print(vocab_size)

# 创建numpy二维数组layer_0，将其所有元素初始化为0
layer_0 = np.zeros([1, vocab_size])
print(layer_0.shape)

# 创建查找表(字典格式)并将各单词的索引存储其中
word2index = {}
for i, word in enumerate(vocab):
    word2index[word] = i

# print(word2index)

# 创建输入数据
# 函数——统计每个单词出现在给定评论里的次数，并将统计结果存入layer_0相应的索引下
def update_input_layer(review):
    global layer_0
    layer_0 *= 0   # 清除之前的状态

    for word in review.split(" "):
        layer_0[0][word2index[word]] += 1


# 测试用第一条评论更新输入层的效果
update_input_layer(reviews[0])
print(layer_0)

# 创建输出数据
# 函数——根据给定标签是NEGATIVE(负面)还是POSITIVE(正面)返回0或1
def get_target_for_label(label):
    if label == 'NEGATIVE':
        return 0
    else:
        return 1


# 测试
print(labels[0])
print(get_target_for_label(labels[0]))

# ----------------------------------------------------------------------------------------------------------------------

"""获得建立好的神经网络模型并利用除最后一千条外的所有评论进行训练，采用学习率0.1(根据训练情况调整)"""
from sentiment_network import *
mlp = SentimentNetwork(reviews[:-1000], labels[:-1000], learning_rate=0.1, min_count=20, polarity_cutoff=0.8)

# 开始训练网络
mlp.train(reviews[:-1000], labels[:-1000])
# 用剩下一千条评论测试网络的性能
mlp.test(reviews[-1000:], labels[-1000:])
