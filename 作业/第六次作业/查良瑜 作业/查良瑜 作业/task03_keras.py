import numpy as np
import keras
from keras import optimizers
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)
# print(np.shape(x_train),np.shape(x_test))
tokenizer = Tokenizer(num_words=1000)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()    #网络结构是1000*512*60*2
model.add(Dense(100, activation='sigmoid', input_dim=1000))
model.add(Dropout(0.5))
model.add(Dense(60,activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])
hist = model.fit(x_train, y_train,
          batch_size=100,
          epochs=40,
          validation_data=(x_test, y_test),
          verbose=2)
score = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: ", score[1])