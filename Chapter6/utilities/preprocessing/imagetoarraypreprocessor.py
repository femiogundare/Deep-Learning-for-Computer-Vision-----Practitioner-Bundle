# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 19:05:43 2020

@author: femiogundare
"""

from keras.preprocessing.image import img_to_array


class ImageToArrayPreprocessor:
    def __init__(self, data_format=None):
        # Store the image data format
        self.data_format = data_format

    def preprocess(self, image):
        # Apply the Keras utility function that correctly rearranges the dimensions of the image
        return img_to_array(image, data_format=self.data_format)
