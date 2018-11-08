import numpy as np
import pandas as pd
"""
@description: 用感知机实现类似逻辑电路中的与门；
              实际上是对于一组训练数据[逻辑电路中的真值表]，其包含x1,x2与y的值[每个变量都只有0,1两种取值]，
              确定合适的参数w1,w2和θ[或w1,w2和b]，使“或”真值表中的逻辑成立，即获得了感知机模型
"""


"""OR函数实现或门：其接收输入信号x1,x2的值，人工定义w1,w2和b的值，通过numpy矩阵运算后判断输出为0还是1"""
def OR(x1, x2):
    x = np.array([x1,x2])  # x数组保存两个输入信号的值
    w = np.array([0.5, 0.5])  # w数组保存对应的两个权重值[人工定义]
    b = -0.2  # b为偏置值，即神经元被激活的容易程度
    result = np.sum(w*x) + b  # 获得感知机模型
    if result <= 0:
        return 0   # 输出值为0
    else:
        return 1   # 输出值为1


"""测试函数"""
def _main():
    y1 = OR(0, 0)
    y2 = OR(1, 0)
    y3 = OR(0, 1)
    y4 = OR(1, 1)

    value_list = [[0, 0, y1], [1, 0, y2], [0, 1, y3], [1, 1, y4]]  # 对应不同x1,x2值的y值列表

    # 指定索引，将列表转化为DataFrame结构
    true_value_table = pd.DataFrame(value_list, index=['case1', 'case2', 'case3', 'case4'], columns=['x1', 'x2', 'y'])
    print("OR函数的真值表：")
    print(true_value_table)


_main()