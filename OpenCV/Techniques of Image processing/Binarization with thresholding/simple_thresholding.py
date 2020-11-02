# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 22:10:42 2020

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

#binarize the image using thresholding
(T, binarized_image)=cv2.threshold(image1, 60, 255, cv2.THRESH_BINARY)
plt.imshow(image1, cmap='gray')

#binarization with inverse thresholding
(Ti, inverse_binarized_image)=cv2.threshold(image1, 60, 255,
                                            cv2.THRESH_BINARY_INV)
plt.imshow(inverse_binarized_image, cmap='gray')


