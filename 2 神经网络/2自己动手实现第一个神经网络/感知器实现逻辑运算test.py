import pandas as pd
#用最简单的神经网络感知器实现逻辑 与、或、非的功能



# 测试与逻辑
weight1 =0.5
weight2 = 0.5
bias =-1.0

# DON'T CHANGE ANYTHING BELOW
# Inputs and outputs
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [False, False, False, True]
outputs = []

# Generate and check output 通过感知器得到输出，并和correct_outputs比较，记录在is_correct_string
for test_input, correct_output in zip(test_inputs, correct_outputs):
    linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
    output = int(linear_combination >= 0)
    is_correct_string = 'Yes' if output == correct_output else 'No'
    outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

# Print output
num_wrong = len([output[4] for output in outputs if output[4] == 'No'])#判断是否有错误次数
output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Activation Output', '  Is Correct'])
if not num_wrong:
    print('Nice!  You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))

import pandas as pd




# 测试亦或逻辑，单隐层神经元，包含两个隐单元
weight111 = 1
weight112 = 1
weight121 = -1
weight122 = -1

weight21 = 1
weight22 = 1

bias11 = -0.5
bias12 = 1.5
bias2 = -1.5

# DON'T CHANGE ANYTHING BELOW
# Inputs and outputs
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [False, True, True, False]
outputs = []

# Generate and check output
for test_input, correct_output in zip(test_inputs, correct_outputs):
    linear_combination11 = weight111 * test_input[0] + weight112 * test_input[1] + bias11
    output1 = int(linear_combination11 >= 0)
    linear_combination12 = weight121 * test_input[0] + weight122 * test_input[1] + bias12
    output2 = int(linear_combination12 >= 0)

    linear_combination12 = weight21 * output1 + weight22 * output2 + bias2

    output = int(linear_combination12 >= 0)
    is_correct_string = 'Yes' if output == correct_output else 'No'
    outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

# Print output
num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Activation Output',
                                              '  Is Correct'])
if not num_wrong:
    print('Nice!  You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))
