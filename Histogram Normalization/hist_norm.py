# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 15:19:39 2022

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



# Path to the training dataset
train_path = r"C:\Users\gjy3r6\Documents\Datasets\hist_norm_datasets\sample_dataset_hist\train\eyeDetected_sure"

# Histogram normalization
train_datagen = ImageDataGenerator(
    featurewise_center = True,
    samplewise_center = False,
    featurewise_std_normalization = False, # across all features
    samplewise_std_normalization = False, # across all samples
    zca_whitening = False,
    zca_epsilon = 1e-06,
    rotation_range = 0,
    width_shift_range = 0.0,
    height_shift_range = 0.0,
    brightness_range = None,
    shear_range = 0.0,
    zoom_range = 0.0,
    channel_shift_range = 0.0,
    fill_mode = 'nearest',
    cval = 0.0,
    horizontal_flip = False,
    vertical_flip = False,
    rescale = None,
    preprocessing_function = None,
    data_format = None,
    validation_split = 0.0,
    dtype = None
    )

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size = (56, 64),
    batch_size = 8, 
    class_mode = "categorical",
    color_mode = "rgb",
    classes = None,
    shuffle = False,
    seed = None,
    save_to_dir = r"C:\Users\gjy3r6\Documents\Datasets\hist_norm_datasets\images\sample_open_sure", # path to save augmented images
    save_prefix = "aug",
    save_format = "jpg",
    follow_links = False, 
    subset = None,
    interpolation = "nearest"
    )

i = 0
for batch in train_generator:
    i += 1
    if i > 4.5: # divide total samples by batch size
        break
print("Augmented images saved!")










