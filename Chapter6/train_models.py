# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 00:23:24 2020

@author: femiogundare
"""


#Import the required libraries
import os
import argparse
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from utilities.nn.cnn.minivggnet import MiniVGGNet

#set matplotlib backend so figures can be saved to the background
matplotlib.use('Agg')

#construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True, 
                help='path to output directory')
ap.add_argument('-m', '--models', required=True, 
                help='path to output models directory')
ap.add_argument('-n', '--num_models', required=True, type=int, default=5,
                help='no.of models to train')
args = vars(ap.parse_args())



(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float') / 255.0   #scale the data to range [0-1] 
X_test = X_test.astype('float') / 255.0    #scale the data to range [0-1]


#convert labels to vectors
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)


labelNames = [
        "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", 
        "ship", "truck"
        ]


#construct image generator for data augmentation
aug = ImageDataGenerator(
        rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, 
        horizontal_flip=True, fill_mode='nearest'
        )

#loop over the number of models to train
for i in range(args['num_models']): 
    print('Training model {}/{}'.format(i+1, args['num_models']))
    model = MiniVGGNet.build(width=32, height=32, depth=3, classes=len(labelNames))
    model.compile(
            optimizer=SGD(learning_rate=0.01, decay=0.01/40, momentum=0.9, nesterov=True), 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
            )
    
    H = model.fit_generator(
            aug.flow(X_train, y_train, batch_size=64), validation_data=(X_test, y_test),
            steps_per_epoch=len(X_train) // 64, epochs=40, verbose=1
            )
    
    #save the model to disk
    p = [args['model'], 'model_{}.model'.format(i)]
    model.save(os.path.sep.join(p))
    
    #evaluate the model
    predictions = model.predict(X_test, batch_size=64)
    class_report = classification_report(
            np.argmax(predictions, axis=1), np.argmax(y_test, axis=1), target_names=lb.classes_
            )
    
    #save the report to file
    q = [args['model'], 'model_{}.txt'.format(i)]
    file = os.path.sep.join(q)
    f = open(file, 'w')
    f.write(class_report)
    f.close()
    
    #plots-------
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, 40), H.history['loss'], label='train_loss')
    plt.plot(np.arange(0, 40), H.history['val_loss'], label='val_loss')
    plt.plot(np.arange(0, 40), H.history['accuracy'], label='train_acc')
    plt.plot(np.arange(0, 40), H.history["val_accuracy"], label='val_acc')
    plt.title("Training Loss and Accuracy for model {}".format(i))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    
    r = [args['model'], '{model}_{}.png'.format(i)]
    plt.savefig(os.path.sep.join(r))
    plt.close()
    
    
    
    
"""
To run:
    python train_models.py --output output --models models
"""