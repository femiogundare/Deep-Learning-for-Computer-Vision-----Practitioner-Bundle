# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 01:17:48 2020

@author: femiogundare
"""


import os
import argparse
import numpy as np
import glob
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.datasets import cifar10
from keras.models import load_model

#construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--models', required=True, help='path to models directory')
args = vars(ap.parse_args())


#load the data and scale to range [0, 1]
X_test, y_test = cifar10.load_data()[1]
X_test = X_test.astype('float') / 255.0

labelNames = [
        "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", 
        "ship", "truck"
        ]

#convert the labels from integers to vectors
lb = LabelBinarizer()
y_test = lb.fit_transform(y_test)


modelPaths = os.path.sep.join([args['models'], '*.model'])
modelPaths = list(glob.glob(modelPaths))
models = []


#loop over the model paths, load the models and append them to the models list
for i, modelPath in enumerate(modelPaths):
    print('Loading model {}/{}'.format(i+1, len(modelPaths)))
    models.append(load_model(modelPath))
    
    
print('Evaluating the ensemble...')
predictions = []

for model in models:
    predictions.append(model.predict(X_test, batch_size=64))
    
    
predictions = np.average(predictions, axis=0)
print(classification_report(
        np.argmax(predictions, axis=1), np.argmax(y_test, axis=1), target_names=labelNames
        )
    )
    
    
#plots
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




"""
To run:
    python test_ensemble.py --models models
"""