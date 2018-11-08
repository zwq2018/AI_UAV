import numpy as np
import pandas as pd
"""
@description: 用感知机实现类似逻辑电路中的与非门；[同时非门的实现即令一个x值的权重为0，从而只根据另一个输入判断输出]
              实际上是对于一组训练数据[逻辑电路中的真值表]，其包含x1,x2与y的值[每个变量都只有0,1两种取值]，
              确定合适的参数w1,w2和b，使“与非”真值表中的逻辑成立，即获得了感知机模型
"""


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


"""测试函数"""
def _main():
    y1 = NAND(0, 0)
    y2 = NAND(1, 0)
    y3 = NAND(0, 1)
    y4 = NAND(1, 1)

    value_list = [[0, 0, y1], [1, 0, y2], [0, 1, y3], [1, 1, y4]]

    true_value_table = pd.DataFrame(value_list, index=['case1', 'case2', 'case3', 'case4'], columns=['x1', 'x2', 'y'])

    print("NAND函数的真值表：")
    print(true_value_table)


_main()