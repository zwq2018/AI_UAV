from load_data import load_train_images,load_train_labels,load_test_images,load_test_labels
from BP_Neural_Network import*
import  matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def main():
    train_images = load_train_images()
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()
    # fig = plt.figure()
    # plt.imshow(test_images[0],cmap = 'Greys',interpolation = 'None')
    # plt.show()
    # 对测试样本和训练样本中的像素值作归一化处理
    train_images_scaled = (train_images/255.0*0.99)+0.01
    test_images_scaled = (test_images/255.0*0.99)+0.01
    input_nodes = 784  # 784?
    output_nodes = 10
    hidden_nodes = 1000
    learningrate = 0.1
    # func_type = 'Relu'
    Cycles = 20
    func_type = 'Sigmoid'
    # Parameters:inputnodes,hiddennodes,outputnodes,learningrate,func_type
    Bp_NN_test = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learningrate,func_type)
    #训练网络
    for i in range(Cycles):
        print("第{}轮训练开始".format(i+1))
        k = 1
        for sample,label in zip(train_images_scaled,train_labels):
            inputs = sample.reshape(-1,1)
            targets = (np.zeros(output_nodes) + 0.01).reshape(-1,1)
            targets[int(label)] = 0.99
            Bp_NN_test.train(inputs,targets)
            if k%1000 == 0:
                print("第{}个样本训练完成".format(k))
            k += 1
        print("第{}轮训练完成".format(i+1))
    # 测试网络
    score = []
    acc = 0.0
    for sample,correct_label in zip(test_images_scaled,test_labels):
        inputs = sample.reshape(-1,1)
        outputs = Bp_NN_test.query(inputs)
        label = np.argmax(outputs)
        print(label, correct_label)
        if label == correct_label:
            score.append(1.0)
        else:
            score.append(0.0)
    acc = np.sum(score)/len(score)
    print("预测准确率是{}".format(acc))
if __name__ == '__main__':
    main()



