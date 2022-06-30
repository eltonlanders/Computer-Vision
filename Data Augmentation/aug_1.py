# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:28:10 2022

@author: gjy3r6 (Elton Landers)
"""
import os
import cv2
import time
import pandas as pd
from glob import glob
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.metrics import confusion_matrix
from sklearn.metrics import *
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model



# Train data source path (images to be augmented)
train_path = r" "

# Data Augmentation
train_datagen = ImageDataGenerator(
    # featurewise_center = False,
    # samplewise_center = False,
    # featurewise_std_normalization = False,
    # samplewise_std_normalization = False,
    # zca_whitening = False,
    # zca_epsilon = 1e-06, # default value is 1e-06
    # rotation_range = 0.0, # randomly in +angle and -angle
    # width_shift_range = 0.0,
    # height_shift_range = 0.0,
    # brightness_range = (0.7, 3.7), # tuple of floats. Roughly preserves the saturation
    # brightness_range = None,
    # shear_range = 0.0,
    # zoom_range = 0.0, # out and in 
    # channel_shift_range = 0, # roughly preserves the contrast
    # fill_mode = 'nearest', # mostly nearest and constant
    # cval = 0.0, # constant (fill_mode) color value
    # horizontal_flip = False,
    # vertical_flip = False,
    # rescale = None, 
    # preprocessing_function = None,
    # data_format = None,
    # validation_split = 0.0,
    # dtype = None
    )


# Data Ingestion 
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size = (504, 656), # height first then width
    batch_size = 1,
    class_mode = "categorical",
    color_mode = "grayscale",
    classes = None,
    shuffle = False, # changed to false 
    seed = None,
    save_to_dir = r" ", # augmented images will be saved here
    save_prefix = "",
    save_format = "jpg",
    follow_links = False, 
    subset = None,
    interpolation = "nearest"
    )


# train_generator.next

i = 0
for batch in train_generator:
    i += 1
    if i > 9182: # number of images to generate irrespective of how many inputs are there
        break
print("Augmented images saved!")
