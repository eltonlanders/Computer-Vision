# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 07:32:20 2020

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
#image1=cv2.bilateralFilter(image1, 5, 50, 50)
plt.imshow(image1, cmap='gray')

#canny function for edge detection
canny=cv2.Canny(image1, 50, 170)
plt.imshow(canny)




