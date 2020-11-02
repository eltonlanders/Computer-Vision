# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 07:08:59 2020

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

#Binarize the image using thresholding
(T, binarizedImage) = cv2.threshold(image1, 0, 255,
                                    cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print("Threshold value with Otsu binarization", T)
plt.imshow(binarizedImage)

#Binarization with inverse thresholding
(Ti, inverseBinarizedImage) = cv2.threshold(image1,
                                            0,
                                            255,
                                            cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
print("Threshold value with Otsu inverse binazarion", Ti)
plt.imshow(inverseBinarizedImage)
