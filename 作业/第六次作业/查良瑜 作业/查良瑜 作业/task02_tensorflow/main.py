from CNN_Tensorflow import*
import pickle

pickle_file = r'\task02_tensorflow\notMNIST.pickle'
with open(pickle_file, 'rb') as f:
  pickle_data = pickle.load(f)
  train_features = pickle_data['train_dataset']
  train_labels = pickle_data['train_labels']
  valid_features = pickle_data['valid_dataset']
  valid_labels = pickle_data['valid_labels']
  test_features = pickle_data['test_dataset']
  test_labels = pickle_data['test_labels']
  del pickle_data  # Free up memory

print('Data and modules loaded.')
# print(np.shape(train_features[0]))
#定义超参数
REGULARIZATIONRATE = 0.001
learning_rate = 0.01
batch_size = 128
conv_drop = 0.9
hidden_drop = 0.5
#实例化cnn

MyCNN = CNN(learning_rate=learning_rate,batch_size=batch_size,REGULARIZATIONRATE=REGULARIZATIONRATE)
#测试
MyCNN.train(test_features,test_labels,conv_drop,hidden_drop)

