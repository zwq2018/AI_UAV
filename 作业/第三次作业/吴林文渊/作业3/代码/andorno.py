import pandas as pd
def andway(test_inputs):
    weight1 = 1.0
    weight2 = 1.0
    bias = -1.5
    test_output=[]
    for test_input in test_inputs:#pd类型
        linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
        output=int(linear_combination>=0)
        test_output.append(output)
    return test_output

def orway(test_inputs):
    weight1=1.0
    weight2=1.0
    bias=-0.5
    test_output=[]
    for test_input in test_inputs:
        linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
        output=int(linear_combination>=0)
        test_output.append(output)
    return test_output

def noway(test_inputs):
    weight=-1.0
    test_output=[]
    for test_input in test_inputs:
        linear_combination = weight * test_input[0]
        output=int(linear_combination>=0)
        test_output.append(output)
    return test_output


def xorway(test_inputs):#直接调用前面的and or not
    andanswer=andway(test_inputs)
    oranswer=orway(test_inputs)
    notanswer=noway(zip(andanswer))
    z=zip(notanswer,oranswer)
    answer=andway(z)
    return answer

def xorway_twolayer(test_inputs):
    #and
    weight1_11 = 1.0
    weight1_12 = 1.0
    bias1_1 = -1.5
    #or
    weight1_21=1.0
    weight1_22=1.0
    bias1_2=-0.5
    #第二层的and
    weight2_1 = 1.0
    weight2_2 = 1.0
    bias2 = -1.5
    test_output = []

    for test_input in test_inputs:#两层结构
        #layer1:and
        linear1_1=test_input[0]*weight1_11+test_input[1]*weight1_12+bias1_1
        #layer1:or
        linear1_2=test_input[0]*weight1_21+test_input[1]*weight1_22+bias1_2
        output1_1=int(linear1_1<0)#求反 or
        output1_2=int(linear1_2>=0)
        #第二层
        linear2=output1_1*weight2_1+output1_2*weight2_2+bias2
        output2=int(linear2>=0)
        test_output.append(output2)
    return test_output


def main():#测试
   test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
   '''print(test_inputs)
    output=orway(test_inputs)
    z=zip(test_inputs,output)
    for i,j in z:
        print(i[0],i[1],j)

    test=[1,0,1,0,1]
    input=zip(test)
    output=noway(input)
    z=zip(test,output)
    for i,j in z:
        print(i,j)'''
   answer=xorway_twolayer(test_inputs)
   z=zip(test_inputs,answer)
   for i,j in z:
       print(i[0],i[1],j)


main()