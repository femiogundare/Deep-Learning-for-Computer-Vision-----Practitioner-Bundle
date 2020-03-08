# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 08:43:35 2020

@author: femiogundare
"""


#import the required libraries
import os
import numpy as np
import argparse
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from utilities.nn.cnn.fcheadnet import FCHeadNet
from utilities.datasets.simpledatasetloader import SimpleDatasetLoader
from utilities.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from utilities.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from utilities.preprocessing.simplepreprocessor import SimplePreprocessor


ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, 
                help='path to the dataset')
ap.add_argument('-m', '--model', required=True, 
                help='path to output model')
args = vars(ap.parse_args())


#extract the dataset from the downloded folder
divide = 80
path = args['dataset']

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

#path to images
imagePaths = list(paths.list_images(args['dataset']))

#construct an image generator for the dataset
aug = ImageDataGenerator(
        rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, 
        zoom_range=0.2, shear_range=0.2, horizontal_flip=True, fill_mode='nearest'
        )


#instantiate the preprocessors
aap = AspectAwarePreprocessor(width=224, height=224)
iap = ImageToArrayPreprocessor()

#load the dataset from disk, then scale the raw pixel intensities to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
data, labels = sdl.load(imagePaths, verbose=500)
data = data.astype('float') / 255.0


X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)


#convert the labels from integers to vectors
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)


#load the VGG network, ensure the head FC layers are left off
baseModel = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

#initialize a new head for the network
headModel = FCHeadNet.build(baseModel, classes=len(flower_classes), D=256)

#place the FC model on the base model --- this become the actual model we train
model = Model(inputs=baseModel.input, outputs=headModel)
#'model' above is as though we built a network right from the input layer to the output
#layer------from FCHeadnet, I can see that the output from the baseModel is flattened
#and converted to a fully connected layer----


#loop over all the layers in the base model and freeze them so they will not be updated
#during the training process---cos I'm training only the FC layers
for layer in baseModel.layers:
    layer.trainable = False
    
#OLUWAFEMI EMMANUEL OGUNDARE ----- 08/03/2020
    
print('Compiling model...')
model.compile(
        optimizer=RMSprop(learning_rate=0.001), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
        )

#train the head of the network...all other layers are frozen---this will allow the new
#FC layers to learn
print('Training head...')
H = model.fit_generator(
        aug.flow(X_train, y_train, batch_size=32), validation_data=(X_test, y_test),
        steps_per_epoch=len(X_train) // 32, epochs=25, verbose=1
        )

print('Evaluating...')
predictions = model.predict(X_test, batch_size=32)
print(
      classification_report(
              np.argmax(predictions, axis=1), np.argmax(y_test, axis=1),
              target_names=lb.classes_
              )
      )
      
"""
The punchline of the above is that each and every image is fed forward; the gradients are
only partially backpropagated(backpropagation here starts at the final output and ends 
just after the FC layers---just before the MaxPooling layer 18. All layers of the base
model have been frozen)
"""
      
#------------------------------------------------------------------------------------

"""
Now that the head FC layers have been trained and its gradient have been backpropagated,
I can begin to unfreeze some of the CONV layers of the base model so as to compare the
results gotten when all the base layers are frozen and only the head FC layers are trained
to when some of the base layers and the head FC layers are trained
"""

#unfreezing the base layers from layer 15 to the end(i.e MaxPooling layer 18)
#CONV==>CONV==>CONV==>POOLING ....... head FC layers are after the pooling layer 18
for layer in baseModel.layers[15:]:
    layer.trainable = True
    
#now recompile the model and retrain!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
print('Recompiling the model...')
model.compile(
        optimizer=SGD(learning_rate=0.001), loss='categorical_crossentropy', 
        metrics=['accuracy']
        )

#retrain the model--- this time around, the gradients of both the FC layers and that of
#the last 4 layers of the base model are updated
print('retraining the model...')
h = model.fit_generator(
        aug.flow(X_train, y_train, batch_size=32), validation_data=(X_test, y_test),
        steps_per_epoch=len(X_train) // 32, epochs=100, verbose=1
        )
print('Re-evaluating...')
predictions = model.predict(X_test, batch_size=32)
print(
      classification_report(
              np.argmax(predictions, axis=1), np.argmax(y_test, axis=1),
              target_names=lb.classes_
              )
      )
      
      
      
#serializing the model to disk
model.save(args['model'])



"""
To run:
python finetune_flowers17.py --dataset ../datasets/animals --model flowers17.model
"""