# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 04:21:02 2020

@author: femiogundare
"""

#Importing the required libraries
import os
import numpy as np
import shutil
import argparse
from imutils import paths
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utilities.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from utilities.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from utilities.datasets.simpledatasetloader import SimpleDatasetLoader
from utilities.nn.cnn.minivggnet import MiniVGGNet
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator


#extract the dataset from the downloded folder (this was inspired by a script I saw)
divide = 80
path = r'C:\Users\USER\Desktop\MY TEXTBOOKS\Computer Vision\Practioner Bundle\My Practioner Bundle Codes\datasets\flowers17'

flower_classes = ['Daffodil', 'Snowdrop', 'Lily_Valley', 'Bluebell',
                  'Crocus', 'Iris', 'Tigerlily', 'Tulip',
                  'Fritillary', 'Sunflower', 'Daisy', 'Colts\'s_Foot',
                  'Dandelion', 'Cowslip', 'Buttercup', 'Windflower', 'Pansy'
                  ]

files = os.listdir(path)

for x in files:
    #select only the images
    if '.jpg' in x:
        index = int(x.split('_')[-1].strip('.jpg')) - 1
        className = index // divide
        className = flower_classes[className]
        os.makedirs(path + '/' + className, exist_ok=True)
        shutil.move(path + '/' + x, path + '/' + className + '/' + x)
    


#initialize the image preprocessors
aap = AspectAwarePreprocessor(64, 64)
iap = ImageToArrayPreprocessor()

#load the dataset from disk, then scale the raw pixel intensities to the range [0, 1]
image_paths = list(paths.list_images(r'C:\Users\USER\Desktop\MY TEXTBOOKS\Computer Vision\Practioner Bundle\My Practioner Bundle Codes\datasets\flowers17'))
sdl = SimpleDatasetLoader([aap, iap])
data, labels = sdl.load(image_paths, verbose=500)
data = data.astype('float') / 255.0


#split the dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


#convert the labels from integers to vectors
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)


#construct an image generator for data augmentation
aug = ImageDataGenerator(
        rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, 
        shear_range=0.2, horizontal_flip=True, fill_mode='nearest'
        )


#build the model
model = MiniVGGNet.build(width=64, height=64, depth=3, classes=len(flower_classes))
model.compile(
        loss='categorical_crossentropy', metrics=['accuracy'],
        optimizer=SGD(learning_rate=0.05, momentum=0.9, nesterov=True)
        )


print('Training the netowrk...')
H = model.fit_generator(
        aug.flow(X_train, y_train, batch_size=32), validation_data=(X_test, y_test),
        steps_per_epoch=len(X_train) // 32, epochs=100, verbose=1
        )

print('Evaluating the network...')
predictions = model.predict(X_test, batch_size=32)
print(classification_report(
        np.argmax(predictions, axis=1), np.argmax(y_test, axis=1), target_names=flower_classes)
)


#plot the acciuracy network
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label = "Training loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label = "Validation loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label = "Training accuracy")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label = "Validation accuracy")
plt.title("Training loss and accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()