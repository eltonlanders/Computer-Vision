# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:51:37 2020

@author: elton
"""

import cv2
import skimage.feature as sk
import numpy as np
import matplotlib.pyplot as plt

image1_path=(r'C:/Users/elton/Documents/Computer Vision/Building computer vision applications using ANN/Techniques of image processing/Image arithmetics and bitwise operations/image1.jpg')
image1=cv2.imread(image1_path)
image1=cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
#image1=cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
plt.imshow(image1)

image1 = cv2.resize(image1,(int(image1.shape[0]/5),int(image1.shape[1]/5)))

#HOG calculation
(HOG, hogImage) = sk.hog(image1, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2),
                         visualize=   True, transform_sqrt=True,
                         block_norm="L2-Hys", feature_vector=True)
print("Image Dimension",image1.shape)
print("Feature Vector Dimension:", HOG.shape)

#showing the original and HOG images
plt.imshow(image1)
plt.imshow(hogImage)
                         
