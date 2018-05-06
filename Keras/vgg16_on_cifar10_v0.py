# *This model uses vgg16 architecture with 10 softmax output units.
# *The dataset used is cifar10 dataset with following classes:
# 
#   airplane 
#   automobile 
#   bird 
#   cat 
#   deer 
#   dog 
#   frog 
#   horse 
#   ship 
#   truck



# *Model got 91.63% train accuracy on 2 epochs 
# *Due to limited hardware test accuracy is not checked ye
# *Results on training data:
#  [Loss, accuracy] = [0.2798416886329651, 0.9184400290489196]


# NOTE: Training for longer period of time will improve this model



import pandas as pd
from keras.datasets import cifar10
import keras.backend as k
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

#Downloading train and test data
(X_train_dat, Y_train), (X_test_dat, Y_test) = cifar10.load_data()


#Reshaping images to fit model, as model accepts minimum 48x48 images
X_train = []
from scipy.misc import imresize
for i in range(X_train_dat.shape[0]):
    X_train.append(imresize(X_train_dat[i], [48, 48]))
X_train = np.array(X_train)


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0])
Y_train = ohe.fit_transform(Y_train).toarray()
Y_train = Y_train.reshape(Y_train.shape[0], 1, 1, Y_train.shape[1])




#Preprocessing training data
X_test = []
from scipy.misc import imresize
for i in range(X_test_dat.shape[0]):
    X_test.append(imresize(X_test_dat[i], [48, 48]))
X_test = np.array(X_test)


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0])
Y_test = ohe.fit_transform(Y_test).toarray()
Y_test = Y_test.reshape(Y_test.shape[0], 1, 1, Y_test.shape[1])






#Downloading Model
vgg16_model = keras.applications.vgg16.VGG16(include_top=False, input_shape=(48,48,3))


model = keras.models.Sequential()
for layer in vgg16_model.layers:
    model.add(layer)

#Removing default ouutput layer
model.layers.pop()

#Setting trainable to false
for layer in model.layers:
    layer.trainable = False

#Custom output layer
xn = keras.layers.Dense(10, activation='softmax')
model.add(xn)

model.compile(optimizer="Adam", loss="binary_crossentropy", metrics = ["accuracy"])
model.fit(x=X_train, y= Y_train, batch_size=64, epochs=2)


#Evaluation on test data
model.evaluate(x=X_test, y=Y_test, batch_size=64)










