import pandas as pd

input = [(0, 0), (0, 1), (1, 0), (1, 1)]   #输入数据
and_output = [False, False, False, True]   #与运算的实际分类结果
or_output = [False, True, True, True]      #或运算的实际分类结果
xor_output = [False, True, True, False]    #异或运算的实际分类结果
outputs = []

"""逻辑与"""
def AND(input, and_output):
    weight1 = 1.0
    weight2 = 1.0
    bias = -1.5

    for input, correct_output in zip(input, and_output):
        linear_combination = weight1 * input[0] + weight2 * input[1] + bias
        model_output = int(linear_combination >= 0)  # 激活：将函数输出转化为分类结果
        is_correct = 'Yes' if model_output == correct_output else 'No'
        outputs.append([input[0], input[1], linear_combination, model_output, is_correct])
    return outputs

"""逻辑或"""
def OR(input, or_output):
    weight1 = 1.0
    weight2 = 1.0
    bias = -0.5     #权值均为1时，-1 < bias < 0

    for input, correct_output in zip(input, or_output):
        linear_combination = weight1 * input[0] + weight2 * input[1] + bias
        model_output = int(linear_combination >= 0)  # 激活：将函数输出转化为分类结果
        is_correct = 'Yes' if model_output == correct_output else 'No'
        outputs.append([input[0], input[1], linear_combination, model_output, is_correct])
    return outputs

"""逻辑异或"""
# 逻辑异或可由两层网络实现：
# 第一层包含两个神经元，分别实现逻辑与非和逻辑或的功能
# 第二层包含一个神经元，实现逻辑与的功能
def XOR(input, xor_output):
    """第一层网络逻辑与的参数"""
    layer1_weight1_1 = 1.0
    layer1_weight1_2 = 1.0
    layer1_bias1 = -1.5
    """第一层网络逻辑或的参数"""
    layer1_weight2_1 = 1.0
    layer1_weight2_2 = 1.0
    layer1_bias2 = -0.5
    """第二层网络逻辑与的参数"""
    layer2_weight1 = 1.0
    layer2_weight2 = 1.0
    layer2_bias = -1.5

    for input, correct_output in zip(input, xor_output):
        """第一层第一个神经元实现逻辑与非"""
        layer1_linear_combination1 = layer1_weight1_1 * input[0] + layer1_weight1_2 * input[1] + layer1_bias1
        layer1_output1 = not(int(layer1_linear_combination1 >= 0))

        """第一层第二个神经元实现逻辑或"""
        layer1_linear_combination2 = layer1_weight2_1 * input[0] + layer1_weight2_2 * input[1] + layer1_bias2
        layer1_output2 = int(layer1_linear_combination2 >= 0)

        """第二层的神经元实现逻辑与"""
        layer2_linear_combination = layer2_weight1 * layer1_output1 + layer2_weight2 * layer1_output2 + layer2_bias
        model_output = int(layer2_linear_combination >= 0)

        is_correct = 'Yes' if model_output == correct_output else 'No'
        outputs.append([input[0], input[1], layer2_linear_combination, model_output, is_correct])
    return outputs

"""结果输出"""
def Print(outputs):
    num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
    output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Activation Output', '  Is Correct'])
    if not num_wrong:
        print('Nice!  You got it all correct.\n')
    else:
        print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
    print(output_frame.to_string(index=False))

outputs = XOR(input, xor_output)
Print(outputs)