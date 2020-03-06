# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 19:53:09 2020

@author: femiogundare
"""


#import the required packages
import os
import numpy as np
import progressbar
import argparse
import random
from matplotlib import pyplot as plt
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from utilities.io.hdf5datasetwriter import HDF5DatasetWriter
from keras.preprocessing.image import img_to_array, load_img
from keras.applications import VGG16, imagenet_utils


#argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to the dataset')
ap.add_argument('-o', '--output', required=True, help='path to output HDF5 file')
ap.add_argument('-b', '--batch_size', type=int, default=32, 
                help='batch size of images to be passed into the network')
ap.add_argument('-s', '--buffer_size', type=int, default=1000, help='size of buffer')
args = vars(ap.parse_args())


#batch size
bs = args['batch_size']
buffer_size = args['buffer_size']

print('Loading the images...')
imagePaths = list(paths.list_images(args['dataset']))
random.shuffle(imagePaths)
labels = [p.split(os.path.sep)[-2] for p in imagePaths]

#convert the labels to integers
le = LabelEncoder()
labels = le.fit_transform(labels)


#load the network
model = VGG16(weights='imagenet', include_top=False)

#initialize the hdf5 dataset writer
dataset = HDF5DatasetWriter(
        dims=(len(imagePaths), 512*7*7), outputPath=args['output'], dataKey='features',
        buffSize=buffer_size
        )
dataset.storeClassLabels(le.classes_)


#initialize the progress bar
widgets = [
        "Extracting Features: ", progressbar.Percentage(), " ",progressbar.Bar(), " ", 
        progressbar.ETA()
        ]
pbar = progressbar.ProgressBar(maxval=len(imagePaths),widgets=widgets).start()


#loop over the images in batches, preprocess them and store them in hdf5
for i in np.arange(0, len(imagePaths), bs):
    batchPaths = imagePaths[i:i + bs]
    batchLabels = labels[i:i + bs]
    batchImages = []
    
    for j, imagePath in enumerate(batchPaths):
        #load the image and convert it to 224*224
        image = load_img(imagePath, target_size=(224, 224))
        #convert the image to array
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)
        
        batchImages.append(image)
        
    #stack the batch images vertically so each assumes a shape of (N, 224, 224, 3)
    batchImages = np.vstack(batchImages)
    #make the model perform prediction
    features = model.predict(batchImages, batch_size=bs) #shape becomes (N, 512, 7, 7)
    #flatten the image vectors
    features = features.reshape((features.shape[0], 512*7*7))
        
    #add the features and labels to the hdf5 dataset
    dataset.add(features, batchLabels)
    pbar.update(i)
        
dataset.close()
pbar.finish()


"""
python extract_features.py --dataset datasets/animals --output hdf5_files/animals.hdf5
"""