import numpy as np
from sklearn.datasets import make_classification
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from scipy.special import expit

def sigmoid_forward(z):
    A = expit(z)
    cache = z
    return A,cache
def sigmoid_backword(dA,cache):
    s,cache = sigmoid_forward(cache)
    dz = dA * s * (1.0 - s)
    return dz
def relu_foward(z):
    A = z * (z>0)
    cache = z
    return A,cache
def relu_backward(dA,cache):
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ

class neuralNetwork():
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate,func_type):
        self.inputnodes = inputnodes
        self.hiddennodes = hiddennodes
        self.outputnodes = outputnodes
        self.learningrate = learningrate
         #初始化权重, 令W0 = 1/(n_node_next)^0.5,
         # 权重向量的维度：列数=n_node_pre，行数=n_node_next
        self.W_hin = np.random.normal(0.0,pow(self.hiddennodes,-0.5),(self.hiddennodes,self.inputnodes))
        self.W_hout = np.random.normal(0.0,pow(self.outputnodes,-0.5),(self.outputnodes,self.hiddennodes))
        self.func_type = func_type     #激活函数类别
        pass

    def query(self,inputs):
        # inputs1 = np.array(inputs,ndmin = 2).T
        hidden_inputs = np.dot(self.W_hin,inputs)  #计算到每个隐含层节点的输入
        if self.func_type == 'Sigmoid':
            hidden_outputs,hidden_inputs = sigmoid_forward(hidden_inputs) #计算隐含层节点经过激活函数后的输出
            final_inputs = np.dot(self.W_hout, hidden_outputs)#计算传入到每个输出节点中的输入数据
            final_outputs,final_inputs = sigmoid_forward(final_inputs)   #计算输出节点经过激活函数后的输出
        elif self.func_type == 'Relu':
            hidden_outputs,hidden_inputs = relu_foward(hidden_inputs)
            final_inputs = np.dot(self.W_hout, hidden_outputs)
            final_outputs,final_inputs = relu_foward(final_inputs)
        else:
           raise NameError('Activation Function not exist,Please choose Sigmoid or Relu function')
        return final_outputs

    def train(self,inputs,targets):
        # input = np.array(inputs,ndmin = 2).T
        # target = np.array(targets,ndmin = 2).T
        #前向传播
        hidden_inputs = np.dot(self.W_hin,inputs)
        if self.func_type == 'Sigmoid':
            hidden_outputs,hidden_inputs = sigmoid_forward(hidden_inputs)  #计算隐含层节点经过激活函数后的输出
            final_inputs = np.dot(self.W_hout, hidden_outputs) #计算传入到每个输出节点中的输入数据
            final_outputs,final_inputs = sigmoid_forward(final_inputs)
        elif self.func_type == 'Relu':
            hidden_outputs,hidden_inputs  = relu_foward(hidden_inputs)
            final_inputs = np.dot(self.W_hout, hidden_outputs)
            final_outputs,final_inputs = relu_foward(final_inputs)
        #误差计算
        output_error = targets - final_outputs          #为什么不能用误差的平方？
        hidden_error_out = np.dot(self.W_hout.T,output_error)
        #梯度下降，dw = -lr * ek * active_func_backward(out) * out
        # self.W_hout += self.learningrate * np.dot(output_error * final_outputs * \
        #                                          (1.0 - final_outputs), np.transpose(hidden_outputs))
        # self.W_hin += self.learningrate * np.dot(hidden_error_out * hidden_outputs * \
        #                                          (1.0 - hidden_outputs), np.transpose(inputs))

        if self.func_type == 'Sigmoid':
            self.W_hout += self.learningrate * np.dot(output_error * \
                                                    sigmoid_backword(final_outputs,final_inputs), np.transpose(hidden_outputs))
            self.W_hin += self.learningrate * np.dot(hidden_error_out * \
                                                    sigmoid_backword(hidden_outputs,hidden_inputs),np.transpose(inputs))
        elif self.func_type == 'Relu':
            self.W_hout += self.learningrate * np.dot(output_error * \
                                                     relu_backward(final_outputs,final_inputs),np.transpose(hidden_outputs))
            self.W_hin += self.learningrate * np.dot(hidden_error_out * \
                                                    relu_backward(hidden_outputs,hidden_inputs),np.transpose(inputs))
        pass









