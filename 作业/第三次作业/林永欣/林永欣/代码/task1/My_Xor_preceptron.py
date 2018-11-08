from pandas import DataFrame

and_weight1 = 0.5
and_weight2 = 0.5
and_bias = -0.7
or_weight1 = 0.8
or_weight2 = 0.8
or_bias = -0.5
not_weight1 = 0
not_weight2 = -1
not_bias = 0.5


# 正确输入和输出
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [False, True, True, False]
outputs = []


def xor_perceptron():
    for test_input, correct_output in zip(test_inputs, correct_outputs):  # zip来融合数据
        and_linear_combination = and_weight1 * test_input[0] + and_weight2 * test_input[1] + and_bias  # 先与后非求出第一个值
        and_output = int(and_linear_combination >= 0)
        and_not_linear_combination = not_weight2 * and_output + not_bias  # 非运算
        and_not_output = int(and_not_linear_combination >= 0)

        or_linear_combination = or_weight1 * test_input[0] + or_weight2 * test_input[1] + or_bias  # 或运算求出第二个值
        or_output = int(or_linear_combination >= 0)

        xor_liner_combination = and_weight1 * and_not_output + and_weight2 * or_output + and_bias  # 对两个值进行与得出异或的结果
        xor_output = int(xor_liner_combination >= 0)

        is_correct_string = 'Yes' if xor_output == correct_output else 'No'
        outputs.append([test_input[0], test_input[1], xor_liner_combination, xor_output, is_correct_string])


def main():
    xor_perceptron()
    wrong_count = len([output[4] for output in outputs if output[4] == 'No'])  # 统计list有多少元素可用len
    output_frame = DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination',
                                                  '  Activation Output', '  Is Correct'])
    if not wrong_count:
        print('Nice!  You got it all correct.\n')
    else:
        print('You got {} wrong.  Keep trying!\n'.format(wrong_count))  # 在指定位置输出的格式用format
    print(output_frame.to_string(index=False))  # 转成字符串后输出


if __name__ == '__main__':
    main()
