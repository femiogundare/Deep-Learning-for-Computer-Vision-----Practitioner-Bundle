# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 06:41:38 2020

@author: femiogundare
"""


#import the required libraries
import argparse
from keras.applications import VGG16


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--include_top', type=int, default=1, 
                help='whether or not to include top CNN')
args = vars(ap.parse_args())


print('Loading the network...')
model = VGG16(
        weights='imagenet', include_top=args['include_top']>0
        )

print('Showing layers...')
for i, layer in enumerate(model.layers):
    print('{} \t{}'.format(i, layer.__class__.__name__))