# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import SGD
from keras.models import load_model
from keras import backend as t
import tensorflow as tf
from keras.models import model_from_json


t.set_image_dim_ordering('th')

seed=1234
np.random.seed(seed)


img_rows = 28
img_cols = 28
epochs = 5
batch_size = 100
drop_rate = 0.4
learn_rate = 0.001


#load the MNIST dataset
##x_train is train_features
##y_train is train_labels it shows number of classes
(x_train,y_train),(x_test,y_test) = mnist.load_data()
image_shape=(1,img_rows,img_cols)


#y_train = odd_even(y_train)
#y_test = odd_even(y_test)

#number of classes
number_of_classes = 2 

# plot images as gray scale
plt.subplot(221)
plt.imshow(x_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(x_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(x_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(x_train[3], cmap=plt.get_cmap('gray'))
plt.show()



x_train = x_train.reshape(x_train.shape[0],1,img_rows,img_cols)
x_test = x_test.reshape(x_test.shape[0],1,img_rows,img_cols)


#to reduce memory requirement
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


#normalize input from 0 to 255 to 0 to 1
x_train = x_train/255
x_test = x_test/255

print(y_train)

#0 represent as even and 1 represent as odd
print((y_train%2!=0).astype(int))

#to convert vector to binary class matrice of odd and even
y_train = np_utils.to_categorical(y_train%2!=0).astype(int)
y_test = np_utils.to_categorical(y_test%2!=0).astype(int)

print(y_train)

def as_keras_metric(method):
    import functools
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        t.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

precision = as_keras_metric(tf.metrics.precision)
recall = as_keras_metric(tf.metrics.recall)

#define model 
def model_cnn():
    model = Sequential()
    model.add(Conv2D(32,kernel_size=(5,5),input_shape=(1, img_rows,img_cols),activation = "relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64,kernel_size=(5,5),activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(drop_rate))
    model.add(Flatten())#convert the 2d matrix into 1d vector
    model.add(Dense(64,activation="relu"))
    model.add(Dense(20,activation ="relu"))
    model.add(Dense(number_of_classes,activation="softmax"))
    return(model)



#calling the model
model = model_cnn()


#compile model
sgd = SGD(lr=0.001)
model.compile(loss="categorical_crossentropy",optimizer=sgd,metrics=["accuracy",precision,recall])
tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
#callbacks=[tensor_board]
#fit the model
#model1 = model.fit(x_train,y_train,validation_split=0.075,epochs=epochs,batch_size=batch_size,verbose=1)
#print(model.history)
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=epochs,batch_size=batch_size,verbose=1,callbacks=[tensor_board])

#evaluate function
value = model.evaluate(x_train,y_train,verbose=0)
print("Total Loss of the model:",value[0])
print("Total Accuracy of the model:",value[1])
#print("Total Precision of the model:",value[2])
#print("Total Recall of the model:",value[3])


#save the model
#model.save('model.h5')

#save the weights
# serialize model to JSON
model_json = model.to_json()
with open("model_sgd.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_sgd.h5")
print("Saved model to disk")
model.save_weights('model_sgd.h5')

#load the model
#load_model('model.h5')

#load_model('model.h5')