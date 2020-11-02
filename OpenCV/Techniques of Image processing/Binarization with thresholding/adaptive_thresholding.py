# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 06:57:18 2020

@author: elton
"""

from __future__ import print_function
import cv2
import matplotlib.pyplot as plt
import numpy as np

image1_path=(r'C:/Users/elton/Documents/Computer Vision/Building computer vision applications using ANN/Techniques of image processing/Image arithmetics and bitwise operations/image1.jpg')
image1=cv2.imread(image1_path)
image1=cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image1=cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
plt.imshow(image1, cmap='gray')

#Binarization using adaptive thresholding and simple mean
binarized=cv2.adaptiveThreshold(image1, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 7, 3)
plt.imshow(binarized)

#Binarization using adaptive thresholding and Gaussian Mean
binarized2=cv2.adaptiveThreshold(image1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 3)
plt.imshow(binarized2)
