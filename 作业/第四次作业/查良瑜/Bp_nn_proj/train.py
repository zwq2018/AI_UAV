from task2_data_processing import*
import tensorflow as tf
from BP_Network_By_Tensorflow import*

Samples_and_targets = data_process()
train_features = Samples_and_targets['train_features']
train_targets = Samples_and_targets['train_targets']

INPUT_NODE = 15 # 输入样本特征维度为56
# OUTPUT_NODE = 3 # 输出targets维度为3
OUTPUT_NODE = 1 # 输出targets维度为1
LAYER1_NODE = 10 # 隐含层节点数
BATCH_SIZE = 7
LEARNING_RATE_BASE = 0.1 # 基学习率
LEARNING_RATE_DECAY = 0.99 # 学习率的衰减率
REGULARIZATION_RATE = 0.0001 # 正则化项的权重系数
MOVING_AVERAGE_DECAY = 0.99 # 滑动平均的衰减系数
TRAINING_STEPS = 20000 #训练步数



#(inputs,targets,REGULARIZATION_RATE,BATCH_SIZE,INPUT_NODE,LAYER1_NODE,OUTPUT_NODE,
                 # MOVING_AVERAGE_DECAY,LEARNING_RATE_BASE,LEARNING_RATE_DECAY)
My_Bp_nn = Bp_Neural_NetworK(REGULARIZATION_RATE,BATCH_SIZE,INPUT_NODE,LAYER1_NODE,
                             OUTPUT_NODE,MOVING_AVERAGE_DECAY,LEARNING_RATE_BASE,LEARNING_RATE_DECAY)

loss = My_Bp_nn.Train(TRAINING_STEPS,train_features,train_targets,isTrain=True)

# for i in range(TRAINING_ITERS):
#     loss = My_Bp_nn.Train()
#     if i % 2 == 0:
#         print("Iters"+str(i)+",loss of model:{}".format(loss))
# print("Optimizer Finished~")







