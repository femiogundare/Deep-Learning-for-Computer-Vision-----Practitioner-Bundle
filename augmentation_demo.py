# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 23:25:40 2020

@author: femiogundare
"""


from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np


#load the desired image, convert it to a numpy array, and then add an extra dimension
image = load_img(r'C:\Users\USER\Pictures\cute_little_dog.jpg')
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

#construct an image generator and then initialize the total number of images generated so far
aug = ImageDataGenerator(
        rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,
        zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
        )
total = 0


#construct the actual image generator and save the generated images to a folder
imageGen = aug.flow(
        image, batch_size=1, save_to_dir=r'C:\Users\USER\Desktop\MY TEXTBOOKS\Computer Vision\Practioner Bundle\My Practioner Bundle Codes\augmentation_demo_images',
        save_prefix='augmentation_image', save_format='jpg'
        )


for image in imageGen:
    total +=1
    
    if total ==10:
        break