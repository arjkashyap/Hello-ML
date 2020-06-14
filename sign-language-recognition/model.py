#!/usr/bin/env python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
from sklearn.model_selection import train_test_split
import time

#Model name for tensor board
NAME = "Sign-Language-cnn-64x128x256-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

X = np.load('features.npy')
y = np.load('labels.npy')
X = X/255.0      # Normalize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.32)

print("Features shape: ", X.shape)
print("Train data size: ", len(X_train))
print("Test Data size: ", len(X_test))


# CNN Model
def cnn_model(X_train, X_test, y_train, y_test):

    model = Sequential()

    # Layer I
    model.add(Conv2D(64,(3,3),activation="relu",input_shape=X.shape[1:]))
    model.add(MaxPooling2D(2,2))

    # Layer II
    #convulation layer 2
    model.add(Conv2D(64,(3,3),activation="relu"))
    model.add(MaxPooling2D(2,2))

    # Flattening the layers
    model.add(Flatten())

    # Fully connected layers
    model.add(Dense(124,activation="relu"))
    model.add(Dense(26,activation="softmax"))

    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data = (X_test, y_test), callbacks = [tensorboard])

    model.save('cnn.model')


cnn_model(X_train, X_test, y_train, y_test)
