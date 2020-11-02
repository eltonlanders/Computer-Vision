# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 07:28:49 2020

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
image1=cv2.bilateralFilter(image1, 5, 50, 50)
plt.imshow(image1, cmap='gray')

#laplace function for edge detection
laplace=cv2.Laplacian(image1, cv2.CV_64F)
laplace=np.uint8(np.absolute(laplace))

plt.imshow(laplace)


