"""
@description  该程序仅是梯度下降法在数学意义上的实现:
            即传入的函数参数是定义的一般性数学函数，传入的x是自变量数组；
            指在学习实现梯度下降法的几个步骤。
"""

import numpy as np

"""
   功能——数值微分求函数的梯度;即求函数对各个自变量求偏导，最终得到各个偏导数组合的向量；
   说明——这里求偏导数时用<数值微分>的方法[即利用“微小的差分”求导数]，需要注意两个方面的问题：
           1. 微小值h的设定。如果h选择地很小（如10e-50），则会产生舍入误差[即因省略小数的精细部分的数值
              而造成最终的计算结果上的误差]；所以这里将微小值h取为1e-4.
           2. 选择计算函数f在哪两者之间的差分。因为h不可能无限接近于0，所以数值微分求到的导数与真的导数（解析导数）
              存在一定的误差。为了减小这个误差，可以选择以x为中心，计算它左右两边的差分（中心差分）。
              即计算f在（x+h）和（x-h）之间的差分。
   参数——f表示函数，x表示一维的自变量数组
"""
def numerical_gradient(f, x):
    h = 1e-4    # 设定微小值h=0.0001
    grad = np.zeros_like(x)    #  生成和x形状相同的默认值全为0的数组用于存梯度值[注意梯度的形状和自变量x的形状相同]

    # 循环求所有x值的梯度
    for idx in range(x.size):
        tmp_val = x[idx]    # tmp_val暂存x中某个分量xi的值

        # 计算f(x+h)
        x[idx] = tmp_val + h    # 只有x[idx]位置的元素改变为xi+h，其余的x值没有变化
        fxh1 = f(x)        #  对x数组中的所有分量求函数值（注意这时x[idx]已经改变）

        # 计算f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        # 求当前遍历到的单个x值的梯度
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val   # 还原该位置的x值

    return grad


"""
   功能——梯度下降法的实现；x值沿着负梯度指示的方向进行更新
   参数——f表示给定的要进行最优化的函数，init_x是自变量x的初始值，lr表示学习率[在多大程度上更新参数]（人工设定为0.01）,
           step_num表示梯度法的重复次数[迭代次数]
"""
def gradient_descent(f, init_x, lr, step_num):
    x = init_x   # 保存初始值

    for i in range(step_num):
        grad = numerical_gradient(f, init_x)  # 求得f对于x的梯度[每一次迭代都要重新求]
        x -= lr * grad

    return x


"""定义一个简单的数学函数"""
def function_simple(x):
    return x[0] ** 2 + x[1] ** 2


"""测试函数"""
def _main():
    """求function_simple函数在x0=3.0,x1=4.0处的梯度"""
    grad = numerical_gradient(function_simple, np.array([3.0, 4.0]))
    print("function_simple在x0=3.0,x1=4.0处的梯度为：")
    print(grad)
    print()

    """用梯度下降法求function_simple函数的最小值"""
    init_x = np.array([-3.0, 4.0])
    x = gradient_descent(function_simple, init_x, lr=0.1, step_num=100)
    print("最小值为：")
    print(x)

_main()