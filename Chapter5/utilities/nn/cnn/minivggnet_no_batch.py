# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 05:25:24 2020

@author: femiogundare
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization, Dropout, Flatten
from keras import backend as K


class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # Initialize the input shape and channel dimension
        input_shape = (height, width, depth)
        channel_dim = -1
        
        # If we are using 'channels_first', update the input shape and channels dimension
        if K.image_data_format() == 'channel_first':
            input_shape = (depth, height, width)
            channel_dim = 1
            
        # Initialize the model
        model = Sequential()
        
        # First CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        #model.add(BatchNormalization(axis=channel_dim))
        
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        #model.add(BatchNormalization(axis=channel_dim))
        
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        model.add(Dropout(0.25))
        
        # Second CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        #model.add(BatchNormalization(axis=channel_dim))
        
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        #model.add(BatchNormalization(axis=channel_dim))
        
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        model.add(Dropout(0.25))
        
        # First (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        #model.add(BatchNormalization(axis=channel_dim))
        model.add(Dropout(0.5))
        
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        
        # Return the constructed network architecture
        return model        