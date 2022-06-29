# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 11:38:29 2021

@author: Elton Landers
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



# Train and Test data paths
train_path = r"" # number of classes
test_path = r""

# start_time = time.time()

# Data Augmentation
train_datagen = ImageDataGenerator(
    featurewise_center = False,
    samplewise_center = False,
    featurewise_std_normalization = False,
    samplewise_std_normalization = False,
    zca_whitening = False,
    zca_epsilon = 1e-06,
    rotation_range = 10,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    brightness_range = None,
    shear_range = 0.1,
    zoom_range = 0.1,
    channel_shift_range = 0.0,
    fill_mode = 'nearest',
    cval = 0.0,
    horizontal_flip = False,
    vertical_flip = False,
    rescale = 1/255,
    preprocessing_function = None,
    data_format = None,
    validation_split = 0.0,
    dtype = None
    )

test_datagen = ImageDataGenerator(
    rescale = 1/255,
    preprocessing_function = None
    )


# Data Ingestion 
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size = (56, 64),
    batch_size = 128,
    class_mode = "categorical",
    color_mode = "grayscale",
    classes = None,
    shuffle = False, # changed to false 
    seed = None,
    save_to_dir = None,
    save_prefix = "",
    save_format = "png",
    follow_links = False, 
    subset = None,
    interpolation = "nearest"
    )

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size = (56, 64),
    batch_size = 128,
    class_mode = "categorical",
    color_mode = "grayscale",
    classes = None,
    shuffle = False, # here too
    seed = None,
    save_to_dir = None,
    save_prefix = "",
    save_format = "png",
    follow_links = False, 
    subset = None,
    interpolation = "nearest"
    )


# image_shape  = (56, 64, 1)

# Initializing the model
model = Sequential()

# First CNN layer
model.add(Conv2D(filters = 32, 
                 kernel_size = (3, 3), 
                 strides = 1, 
                 padding = 'same', 
                 input_shape = (56, 64, 1), # since grayscale
                 activation = 'relu',))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Second CNN Layer
model.add(Conv2D(filters = 64, 
                 kernel_size = (3, 3), 
                 strides = 1, 
                 padding = 'same', 
                 activation = 'relu',))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Third CNN Layer
model.add(Conv2D(filters = 128, 
                 kernel_size = (3, 3), 
                 strides = 1, 
                 padding = 'same', 
                 activation = 'relu',))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(1024))
model.add(Activation('relu'))

# model.add(Dropout(0.5))

# Output layers
model.add(Dense(4))
model.add(Activation('softmax'))

# Compiling the model
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy']
              )

model.summary() 

"""
early_stop = EarlyStopping(monitor = 'val_accuracy', 
                           mode = 'max', 
                           verbose = 1, 
                           patience = 2
                           )
"""

# Save the model using callbacks
checkpoint_dir = "eye_detection_models/eye_detection_01.h5"

checkpoint = ModelCheckpoint(
    filepath = checkpoint_dir,
    # frequency = "epoch",
    save_weights_only = False, # save the entire model not just the weights
    monitor = "val_accuracy",
    mode = "max",
    save_best_only = True, 
    verbose = True
    )

"""
# Learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
"""

# Training the model 
model.fit(
    train_generator,
    epochs = 100,
    validation_data = test_generator
    # callbacks = [early_stop] 
    # callbacks = [checkpoint]
    )

# model.save("model.h5") # manually saving the model after training

# Loading a saved model
# model = load_model(checkpoint_dir) # loading full model
# model = load_weights(checkpoint_filepath) # loading only the weights
# model.predict()


# Getting the predictions
y_pred = model.predict(test_generator)
predictions = np.round(abs(y_pred))
predictions = np.argmax(predictions, axis = 1)
                       
classes = test_generator.classes

# Creating confusion matrix
cm = confusion_matrix(classes, predictions)
# accuracy_score(y_true, np.round(abs(y_pred)), normalize=False) 
precision = precision_score(classes, predictions, average = None)
recall = recall_score(classes, predictions, average = None)

# Creating classification report
classification_report = classification_report(classes, 
                                              predictions, 
                                              digits = 3)

print(cm)


# Plotting the confusion matrix
ls = [] # list of labels

disp = ConfusionMatrixDisplay(
    confusion_matrix = cm, 
    display_labels = ls
    )

disp.plot(
    include_values = True, 
    cmap = "viridis", 
    xticks_rotation = 45,
    values_format = None,
    ax = None,
    colorbar = True
    )

plt.show()


# Plotting some graphs
classification_report = classification_report(classes, 
                                              predictions, 
                                              digits = 3,
                                              output_dict = True, # True when plotting heatmap
                                              target_names = []) # list of labels

report = pd.DataFrame(classification_report).T
report = report.iloc[:-3, :]

plt.xlabel("class labels")
plt.ylabel("# of samples")
plt.title("# of samples across classes")
plt.plot(report['support'],
         color = 'g',
         linestyle = 'solid',
         marker = 'x',
         markerfacecolor = 'yellow',
         markersize = 10, 
         )

"""
losses = pd.DataFrame(model.history.history)
print(losses)
losses.to_csv('model_history.csv', 
              index = False)

losses[['loss','val_loss']].plot()

model.save('..\\models\\eye_detection.h5')
print("--- {:.3f} seconds ---".format(time.time() - start_time))
"""



