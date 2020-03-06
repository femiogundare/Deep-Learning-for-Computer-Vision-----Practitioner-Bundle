# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 00:18:08 2020

@author: femiogundare
"""

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Flatten, Dense
from keras import backend as K


print('Checking certain info about the Keras backend...')
print('Image Data Format: {}'.format(K.image_data_format()))
print('Keras Backend: {}'.format(K.backend()))


class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height, width, depth)
        
        # If we are using 'channels-first', update the input shape
        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)
            
        # Initialize the model
        model = Sequential()
        
        # First set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        # Second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        # First (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))
        
        # Softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        
        # Return the model architecture
        return model