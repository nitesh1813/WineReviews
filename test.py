

from __future__ import print_function
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import main

batch_size = 128
# num_classes = 10
epochs = 20

# the data, shuffled and split between train and test sets
X,y= main.getData()
X=np.array(X)
y=np.array(y)
l=len(X)
x_train=X[:int(0.6*l)]
y_train=y[:int(0.6*l)]
x_val=X[int(0.6*l):int(0.8*l)]
y_val=y[int(0.6*l):int(0.8*l)]
x_test=X[int(0.8*l):]
y_test=y[int(0.8*l):]

# x_train = x_train.reshape(l, 4)
# x_test = x_test.reshape(10000, 784)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

# # convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
# batch_size=100
model = Sequential()
model.add(Dense(64,activation='relu',input_shape=(4,)))
model.add(Dense(64, activation='relu'))
# model.add(Activation('tanh'))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer='rmsprop')


history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_val, y_val))
predict=model.predict(x_test)
# from math import abs
# score = model.evaluate(x_test, y_test, verbose=0)
mse=[abs((x[0]-y)) for x,y in zip(predict,y_test)]
# print (mse[0])
print(sum(mse)*1.0/len(mse))
# print('Test accuracy:', score[1])
