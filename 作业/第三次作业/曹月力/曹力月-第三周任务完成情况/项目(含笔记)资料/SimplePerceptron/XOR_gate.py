import numpy as np
import pandas as pd
"""
感知机的局限性：单层的感知机模型为一条直线，其无法分割非线性的空间，也就无法表示异或门.[用图像的方式观察会更为直观]
解决办法：双层感知机.[试想一条直线不能把空间中不同的点隔离开，就绘制两条直线]
          其实质思想可以联想逻辑电路的组合：想要实现异或逻辑电路，则利用与门，或门，与非门组合的方式可以实现，在逻辑电路图
          中，x1和x2作为输入信号分别传给与非门和或门[这两个门在同一层]，之后与非门和或门的输出作为新的两个输入信号传给与门，
          这样得到的最终输出与异或门的逻辑相同。
          需要明确的是与、或、与非都是相同的单层感知机模型，只是对应的参数w1,w2,b不同而已，所以可以将它们组合成两层感知机
"""


"""这里将之前实现的与、或、与非逻辑函数进行整合，方便实现异或逻辑[调用函数结果即可]"""

"""AND2函数确定w1,w2,b(偏置)的值，且利用numpy库中的矩阵运算方法来实现与门"""
def AND(x1, x2):
    x = np.array([x1, x2])  # 创建输入信号x的numpy数组
    w = np.array([0.5, 0.5])  # 创建权重w的numpy数组，且人工定义值
    b = -0.7  # 偏置b是阈值theta的相反数
    result = np.sum(w*x) + b   # 用numpy库的内置函数实现感知机的模型

    if result <= 0:
        return 0
    else:
        return 1


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


"""NAND函数实现与非门：其接收输入信号x1,x2的值，人工定义w1,w2和b的值，通过numpy矩阵运算后判断输出为0还是1"""
def NAND(x1, x2):
    x = np.array([x1,x2])
    w = np.array([-0.5, -0.5])  # 实际上，与非门参数的确定可以是与门中对应参数的相反数
    b = 0.7
    result = np.sum(w*x) + b

    if result <= 0:
        return 0
    else:
        return 1


"""XOR函数通过双层感知机模型实现异或逻辑，其接收输入x1,x2，传递给中间层"""
def XOR(x1, x2):
    x = np.array([x1, x2])  # x1,x2作为初始的输入信号

    """得到第一层的两个输出"""
    s1 = NAND(x1, x2)    # s1表示与非门的输出[不同x1,x2值的组合有不同的输出]
    s2 = OR(x1, x2)    # s2表示或门的输出

    """第一层的两个输出作为最后一层的输入"""
    y = AND(s1, s2)  # 与门作为最后一层，其输入为上一层的输出s1,s2，输出结果即为最终的异或逻辑

    return y


"""测试函数"""
def _main():
    y1 = XOR(0, 0)
    y2 = XOR(1, 0)
    y3 = XOR(0, 1)
    y4 = XOR(1, 1)

    value_list = [[0, 0, y1], [1, 0, y2], [0, 1, y3], [1, 1, y4]]  # 对应不同x1,x2值的y值列表

    # 指定索引，将列表转化为DataFrame结构
    true_value_table = pd.DataFrame(value_list, index=['case1', 'case2', 'case3', 'case4'], columns=['x1', 'x2', 'y'])

    print("XOR函数的真值表：")
    print(true_value_table)


_main()