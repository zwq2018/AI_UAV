import numpy as np
class NeuralNetwork(object):
    def __init__(self,input_nodes,hidden_nodes,output_nodes,learning_rate):
        #带有self的是class的全局变量 calss的内部函数都可以更改
        self.input_nodes=input_nodes#X的维度
        self.hidden_nodes=hidden_nodes#隐藏层的节点个数
        self.output_nodes=output_nodes#输出层的节点个数

        #初始化各种数据
        self.weights_input_to_hidden= np.random.normal(0.0, self.input_nodes**-0.5, size=(self.input_nodes, self.hidden_nodes)) #输入到隐层
        self.weights_hidden_to_output=np.random.normal(0.0, self.input_nodes**-0.5, size=(self.hidden_nodes, self.output_nodes)) #隐层到输出 此处为[n,m]  非[n,] 注意后面相关矩阵的维度格式

        self.lr=learning_rate

    #激活函数
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    #激活函数的导数
    def sigmoid_prime(self,y):
        return y * (1 - y)

    #前向传播的到各个节点的输出值
    def forward_pass_train(self,x):

        #隐藏节点的输入和输出
        hidden_inputs=np.dot(x,self.weights_input_to_hidden)
        hidden_outputs=self.sigmoid(hidden_inputs)

        #输出节点的输入与输出   输出不需要调整为概率  直接是预测值就可以
        final_inputs=np.dot(hidden_outputs,self.weights_hidden_to_output)
        final_outputs=final_inputs

        return final_inputs,final_outputs,hidden_inputs,hidden_outputs

    # 反向传播对权重的更新
    def backpropagation(self,final_inputs,final_outputs,hidden_inputs,hidden_outputs,x,y,delta_weights_i_h, delta_weights_h_o):
        #误差
        error=y-final_outputs

        #输出层  此层没有使用激活函数,导数为1
        output_error=error*1

        #隐藏层
        hidden_error=output_error*self.weights_hidden_to_output.T*self.sigmoid_prime(hidden_outputs)#此处注意 反向时 因为该权重有完整维度权重的维度也要反  否则无法匹配


        #权重改变累计值
        delta_weights_h_o=delta_weights_h_o+output_error*np.expand_dims(hidden_outputs,axis=1)*self.lr
        delta_weights_i_h=delta_weights_i_h+hidden_error*np.expand_dims(x,axis=1)*self.lr#

        return delta_weights_i_h,delta_weights_h_o

    #权重更新
    def update_weights(self,delta_weights_i_h,delta_weights_h_o,n_records):
        self.weights_hidden_to_output=self.weights_hidden_to_output+self.lr*delta_weights_h_o/n_records
        self.weights_input_to_hidden=self.weights_input_to_hidden+self.lr*delta_weights_i_h/n_records

    # 训练
    def train(self,features,targets):
        n_records,n_feature=features.shape

        #初始化权重累计值
        delta_weights_h_o=np.zeros(self.weights_hidden_to_output.shape)
        delta_weights_i_h=np.zeros(self.weights_input_to_hidden.shape)

        #投放数据
        for x,y in zip(features,targets):
            final_inputs, final_outputs, hidden_inputs, hidden_outputs=self.forward_pass_train(x)#前向传播的输出，和隐藏层的输出
            delta_weights_i_h,delta_weights_h_o=self.backpropagation(final_inputs,final_outputs,hidden_inputs,hidden_outputs,x,y,delta_weights_i_h, delta_weights_h_o)#反向传播，找到权重更新

        self.update_weights(delta_weights_i_h,delta_weights_h_o,n_records)

    #根据最后确定的权重，计算最终输出
    def run(self,features):
        hidden_inputs=np.dot(features,self.weights_input_to_hidden)
        hidden_outputs=self.sigmoid(hidden_inputs)

        final_inputs=np.dot(hidden_outputs,self.weights_hidden_to_output)
        final_outputs=final_inputs

        return final_outputs


