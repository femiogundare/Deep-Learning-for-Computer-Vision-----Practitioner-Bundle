# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 08:18:28 2020

@author: femiogundare
"""


from keras.layers import Flatten, Dense, Dropout


#build the class
class FCHeadNet:
    @staticbuild
    def build(baseModel, classes, D):
        headModel = baseModel.output
        headModel = Flatten(name='flatten')(headModel)
        headModel = Dense(D, activation='relu')(headModel)
        headModel = Dropout(0.5)(headModel)
        
        #add the output layer
        headModel = Dense(classes, activation='softmax')(headModel)
        
        return headModel
    
        """
        this network consists of two FC layers
        """