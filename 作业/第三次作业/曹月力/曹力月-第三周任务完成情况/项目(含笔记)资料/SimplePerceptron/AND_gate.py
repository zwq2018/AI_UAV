import pandas as pd
import numpy as np
"""
@description：用感知机实现类似逻辑电路中的与门；
              实际上是对于一组训练数据[逻辑电路中的真值表]，其包含x1,x2与y的值[每个变量都只有0,1两种取值]，
              确定合适的参数w1,w2和θ[或w1,w2和b]，使“与”真值表中的逻辑成立，即获得了感知机模型
"""
"""ANDi函数实现与门，其接收输入信号x1和x2"""

"""AND1函数确定w1,w2,theta(阈值)的值，利用定义数学公式的方法来实现与门"""
def AND1(x1,x2):
    w1, w2, theta = 0.5, 0.5, 0.7  # 观察真值表，人工定义符合条件的参数的值
    result = x1*w1 + x2*w2  #x1和x2分别乘以各自的权重，将信号总和result传送给神经元
    if result <= theta:
        return 0    # 如果输入信号的总和不超过（小于等于）神经元的阈值，则神经元不被激活，输出为0
    elif result > theta:
        return 1    # 如果输入信号的总和大于神经元的阈值，则神经元被激活，输出为1


"""AND2函数确定w1,w2,b(偏置)的值，且利用numpy库中的矩阵运算方法来实现与门"""
def AND2(x1, x2):
    x = np.array([x1, x2])  # 创建输入信号x的numpy数组
    w = np.array([0.5, 0.5])  # 创建权重w的numpy数组，且人工定义值
    b = -0.7  # 偏置b是阈值theta的相反数
    result = np.sum(w*x) + b   # 用numpy库的内置函数实现感知机的模型

    if result <= 0:
        return 0
    else:
        return 1


"""main函数进行测试"""
def _main():
    y11 = AND1(0, 0)
    y12 = AND1(1, 0)
    y13 = AND1(0, 1)
    y14 = AND1(1, 1)

    value_list1 = [[0, 0, y11], [1, 0, y12], [0, 1, y13], [1, 1, y14]]  # 对应不同x1,x2值的y值列表

    # 指定索引，将列表转化为DataFrame结构
    true_value_table1 = pd.DataFrame(value_list1, index=['case1', 'case2', 'case3', 'case4'], columns=['x1', 'x2', 'y'])
    print("AND1函数的真值表：")
    print(true_value_table1)
    print()

    y21 = AND1(0, 0)
    y22 = AND1(1, 0)
    y23 = AND1(0, 1)
    y24 = AND1(1, 1)

    value_list2 = [[0, 0, y21], [1, 0, y22], [0, 1, y23], [1, 1, y24]]
    true_value_table2 = pd.DataFrame(value_list1, index=['case1', 'case2', 'case3', 'case4'], columns=['x1', 'x2', 'y'])
    print("AND2函数的真值表：")
    print(true_value_table2)

_main()
