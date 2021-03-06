# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 19:07:55 2020

@author: femiogundare
"""
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, Flatten
from keras import backend as K

print('Keras Backend Info: {}'.format(K))

class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        # Initialize the model along with the input shape to be 'channels_last'
        model = Sequential()
        input_shape = (height, width, depth)

        # Update the image shape if 'channels_first' is being used
        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)
        
        # Define the first (and only) CONV => RELU layer
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
       
        # Add a softmax classifier
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        
        return model