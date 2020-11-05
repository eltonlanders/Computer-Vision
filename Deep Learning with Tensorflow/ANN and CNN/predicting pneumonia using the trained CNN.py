import numpy as np
import pathlib
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

model_path = "models/pneumiacnn"

val_img_dir ="images/chest_xray/val"
# ImageDataGenerator class provides mechaism to load both small and large dataset.
# Instruct ImageDataGenerator to scale to normalize pixel values to range (0, 1)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
#Create training image iterator that will be loaded in small batch size. Resize all images to a standard sizes.
val_it = datagen.flow_from_directory(val_img_dir, batch_size=2, target_size=(512,512))


# Load and create the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model(model_path)

# Predict the class of the input image from the loaded model
predicted = model.predict_generator(val_it, steps=24)
print("Predicted", predicted)